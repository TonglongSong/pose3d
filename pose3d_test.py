import time
from rootnetsum import findrootdepth, load_rootnet_data
from posenetsum import load_posenet_data, print_json
from yolo_opencv import get_bbox, load_yolo
import timeit
import paho.mqtt.client as mqtt
import os
import numpy as np
import cv2
from my_utils import gettime
from camera_parameters import get_cam_parameter
from argparse import ArgumentParser
import json
import os.path as osp
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from posenet.main.config import cfg
from posenet.main.model import get_pose_net
from posenet.data.dataset import generate_patch_image
from posenet.common.utils.pose_utils import process_bbox, pixel2cam
from posenet.common.utils.vis import vis_keypoints, vis_3d_multiple_skeleton
from datetime import datetime
import utm


def coordtransform(pose3d, Rvec, Tvec):
    out = np.copy(pose3d)
    for i in range(len(pose3d)):
        for j in range(len(pose3d[i])):
            joint = pose3d[i][j]
            mat = np.matrix(np.array([[joint[0]], [joint[1]], [joint[2]]], dtype=np.float32))
            new = Rvec ** -1 * (mat - Tvec)
            out[i][j] = new.transpose()
    return out


def human_3d_detection(bbox, depth_list, focal, original_img, model, R, T):
    # MuCo joint set
    joint_num = 21
    joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
    skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
        (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))

    # prepare input image
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    bbox_list = bbox  # xmin, ymin, width, height
    root_depth_list = depth_list  # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
    assert len(bbox_list) == len(root_depth_list)
    person_num = len(bbox_list)

    # normalized camera intrinsics
    princpt = [original_img_width / 2, original_img_height / 2]  # x-axis, y-axis


    # for each cropped and resized human image, forward it to PoseNet
    output_pose_2d_list = []
    output_pose_3d_list = []
    for n in range(person_num):
        bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
        img = transform(img).cuda()[None, :, :, :]

        # forward
        with torch.no_grad():
            pose_3d = model(img)  # x,y: pixel, z: root-relative depth (mm)

        # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0].cpu().numpy()
        pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
        pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        output_pose_2d_list.append(pose_3d[:, :2].copy())

        # root-relative discretized depth -> absolute continuous depth
        pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + root_depth_list[n]
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        output_pose_3d_list.append(pose_3d.copy())

    vis_img = original_img.copy()
    for n in range(person_num):
        vis_kps = np.zeros((3, joint_num))
        vis_kps[0, :] = output_pose_2d_list[n][:, 0]
        vis_kps[1, :] = output_pose_2d_list[n][:, 1]
        vis_kps[2, :] = 1
        vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
    cv2.imwrite('output_pose_2d.jpg', vis_img)

    # visualize 3d poses
    Rvec = np.matrix(R)
    Tvec = np.matrix(T)
    vis_kps = np.array(output_pose_3d_list)
    vis_kps = coordtransform(vis_kps, Rvec, Tvec)
    print(print_json(vis_kps))
    vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, vis_img)
    return print_json(vis_kps)

def combined(image, net, classes, rootnet, posenet):
    bbox_list = get_bbox(image, classes, net)
    focal = [args.focal, args.focal]
    depth_list = findrootdepth(bbox_list, focal, image, rootnet)
    pload = human_3d_detection(bbox_list, depth_list, focal, image, posenet, Rvec, Tvec)
    return pload


def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def pose3d(net, classes, rootnet, posenet):
    lastimg = 0
    while True:
        t = gettime() - 30
        if lastimg != t:
            img_path = f"frames/cam{args.camera}/{t}.jpg"
            if os.path.exists(img_path):
                start = timeit.default_timer()
                image = undistort(img_path)
                #image = cv2.imread(img_path)
                pload = combined(image, net, classes, rootnet, posenet)
                stop = timeit.default_timer()
                print(pload)
                print(f"skeleton {t} published in {stop - start} second")
                lastimg = t
            else:
                print(f'waiting for image {t}')
                time.sleep(1)
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--camera', type=int, default=0, help='which camera parameter to be used')
    parser.add_argument(
        '--focal', type=int, default=900, help='focal length')
    args = parser.parse_args()

    # load trained model
    rootnet_model = load_rootnet_data()
    posenet_model = load_posenet_data()
    classes, net = load_yolo()

    # load camera parameters
    DIM = (1920, 1080)
    K, D, Rvec, Tvec = get_cam_parameter(args.camera)
    # start loop
    image = undistort("frames/cam0/123.jpg")
    combined(image, net, classes, rootnet_model, posenet_model)




