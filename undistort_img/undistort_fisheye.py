import numpy as np
import cv2
import glob
import os
import timeit
import time
# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[611.1549814728781, 0.0, 959.5], [0.0, 611.1549814728781, 539.5], [0.0, 0.0, 1.0]])
D=np.array([[0.0], [0.0], [0.0], [0.0]])
img_dir = 'distorted_images'
out_dir = 'undistorted_from_fisheye'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def undistorted(img_path):
    start = timeit.default_timer()
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img_name = img_path.split('\\')[1]
    cv2.imwrite(f"{out_dir}/{img_name}", undistorted_img)
    print(f"{out_dir}/{img_name} undistorted successfully")
    cv2.destroyAllWindows()
    stop = timeit.default_timer()
    print(f"{img_path} undistorted in {stop - start} second")


images = glob.glob(f"{img_dir}/*.jpg")

for fname in images:
    undistorted(fname)

