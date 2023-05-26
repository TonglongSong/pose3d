import cv2
import numpy as np

K=np.array([[915.0385604626166, 0.0, 922.0843377635293], [0.0, 913.6793991039979, 522.6426523782426], [0.0, 0.0, 1.0]])
D=np.array([[0.4289666276805924, -0.1663767172390245, -0.018020650165055854, -0.03482441195465059, 0.22047854218049334]])
obj = np.array([
    [-500, 1500, 0],
    [0, 3500, 0],
    [0, 4500, 0],
    [-500, 3500, 0]
], dtype=np.float64)
img = np.array([
    [1207, 445],
    [824, 497],
    [589, 552],
    [873, 543]
], dtype=np.float64)


def pnpsolve(K, D, obj, img):
    ret, rv, tv = cv2.solvePnP(obj, img, K, D)
    r, _ = cv2.Rodrigues(rv)
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("Rvec=np.array(" + str(r.tolist()) + ")")
    print("Tvec=np.array(" + str(tv.tolist()) + ")")
    campos = -np.dot(np.linalg.inv(r), tv)
    print("camposit"
          "ion =np.array(" + str(campos.tolist()) + ")")

print('For cam0')
print('K=np.array([[611.1549814728781, 0.0, 959.5], [0.0, 611.1549814728781, 539.5], [0.0, 0.0, 1.0]])')
print('D=np.array([[0.0], [0.0], [0.0], [0.0]])')
pnpsolve(K, D, obj, img)


