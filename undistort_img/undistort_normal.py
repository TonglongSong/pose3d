import numpy as np
import cv2
import glob
import os
# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[915.0385604626166, 0.0, 922.0843377635293], [0.0, 913.6793991039979, 522.6426523782426], [0.0, 0.0, 1.0]])
D=np.array([[0.4289666276805924, -0.1663767172390245, -0.018020650165055854, -0.03482441195465059, 0.22047854218049334]])
img_dir = 'undistorted_from_fisheye'
out_dir = 'undistorted'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def undistort(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, K, D)
    img_name = img_path.split('\\')[1]
    cv2.imwrite(f"{out_dir}/{img_name}", dst)
    cv2.destroyAllWindows()


images = glob.glob(f"{img_dir}/*.jpg")

for fname in images:
    undistort(fname)

