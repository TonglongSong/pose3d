import cv2
import numpy as np

K=np.array([[915.0385565530959, 0.0, 922.0843398337839], [0.0, 913.6793963090881, 522.6426489936852], [0.0, 0.0, 1.0]])
D=np.array([[0.4289666292002352, -0.16637673080835746, -0.01802065233170858, -0.03482441066886704, 0.22047855382633968]])
obj = np.array([
    [-1000, 1500, 0],
    [1000, 1500, 0],
    [-1000, 4500, 0],
    [1000, 4500, 0]
], dtype=np.float64)
img = np.array([
    [1268, 669],
    [1013, 551],
    [620, 899],
    [485, 660]
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


