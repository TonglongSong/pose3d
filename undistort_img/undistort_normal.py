import numpy as np
import cv2
import glob
import os
# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[915.0385565530959, 0.0, 922.0843398337839], [0.0, 913.6793963090881, 522.6426489936852], [0.0, 0.0, 1.0]])
D=np.array([[0.4289666292002352, -0.16637673080835746, -0.01802065233170858, -0.03482441066886704, 0.22047855382633968]])
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

