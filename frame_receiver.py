# By Tonglong Song 2022/6/8 17:57
import paramiko
from scp import SCPClient
from my_utils import gettime, clean_history
from argparse import ArgumentParser
import timeit
import time
import os
import cv2
from camera_parameters import get_cam_parameter
import numpy as np


def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def dt_connect():
    k = paramiko.RSAKey.from_private_key_file("app1server.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("connecting")
    c.connect(hostname="115.146.94.74", username="ubuntu", pkey=k)
    print("connected")
    return c


def receiveimg():
    c = dt_connect()
    lastrec = 0
    i = camera
    while True:
        t = gettime() - 20
        if lastrec != t:
            if c.get_transport().is_alive():
                try:
                    with SCPClient(c.get_transport()) as scp:
                        start = timeit.default_timer()
                        scp.get(f"/home/ubuntu/posecapture/tempdata/cam{i}/{t}.jpg", f"frames/cam{i}")
                        undistorted_img = undistort(f"frames/cam{i}/{t}.jpg")
                        cv2.imwrite(f"frames/cam{i}/{t}.jpg", undistorted_img)
                        stop = timeit.default_timer()
                        print(f"{t}.jpg received from camera{i} in {stop - start} second")
                    lastrec = t
                except Exception as e:
                    if os.path.exists(f"frames/cam{i}/{t}.jpg"):
                        os.remove(f"frames/cam{i}/{t}.jpg")
                    print(e)
                    time.sleep(0.5)
            else:
                print("disconnected, trying to reconnect")
                try:
                    c = dt_connect()
                except Exception as e:
                    print(e)
        else:
            time.sleep(0.1)

        # if gettime() % 1000 == 0:
        #     clean_history(300, f"frames/cam{i}")
        #     print('history cleaned')
        #     time.sleep(0.5)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--camera-number', type=int, default="0", help='list of camera id you want to receive image from')
    args = parser.parse_args()
    camera = args.camera_number
    DIM = (1920, 1080)
    K, D, Rvec, Tvec = get_cam_parameter(args.camera_number)
    receiveimg()