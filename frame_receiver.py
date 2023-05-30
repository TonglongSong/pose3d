# By Tonglong Song 2022/6/8 17:57
import paramiko
from scp import SCPClient
from my_utils import gettime, dt_connect
import timeit
import time
import os
import cv2
from config import get_cam_parameter, DT_IMGPATH
import numpy as np


def receiveimg():
    c = dt_connect()
    lastrec = 0
    while True:
        t = gettime() - 20
        if lastrec != t:
            if c.get_transport().is_alive():
                try:
                    with SCPClient(c.get_transport()) as scp:
                        start = timeit.default_timer()
                        scp.get(DT_IMGPATH + f"{t}.jpg", f"frames")
                        undistorted_img = undistort(f"frames/{t}.jpg")
                        cv2.imwrite(f"frames/{t}.jpg", undistorted_img)
                        stop = timeit.default_timer()
                        print(f"{t}.jpg received and undistorted in {stop - start} second")
                    lastrec = t
                except Exception as e:
                    if os.path.exists(f"frames/{t}.jpg"):
                        os.remove(f"frames/{t}.jpg")
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


def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


if __name__ == '__main__':
    K, D, Rvec, Tvec, DIM = get_cam_parameter()
    receiveimg()