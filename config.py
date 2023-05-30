import numpy as np

DT_HOSTNAME = "115.146.94.74"
DT_USERNAME = "ubuntu"
DT_KEYFILEPATH = "app1server.pem"
DT_IMGPATH = '/home/ubuntu/posecapture/tempdata/cam0/'

DT_MQTT_HOST = 'mqtt.digitwin.com.au'
DT_MQTT_USERNAME = 'csdila-3dpose-publisher'
DT_MQTT_PASSWORD = 'ux693dD3x'
DT_MQTT_KEYFILEPATH = 'isrgrootx1.pem'
DT_MQTT_TOPIC = 'uom/parkville/melbourneconnect/level6/csdila/3dpose/humanSkeleton'
DT_MQTT_PORT = 8883

ORIGIN_WGS84 = (144.964506, -37.800165)
REFERENCE_WGS84 = (144.964444, -37.800127)
REFERENCE_LOCAL = (0, 7)


def get_cam_parameter():
    K = np.array([[611.1549814728781, 0.0, 959.5], [0.0, 611.1549814728781, 539.5], [0.0, 0.0, 1.0]])
    D = np.array([[0.0], [0.0], [0.0], [0.0]])
    Rvec = np.array([[-0.5078205930618596, -0.8614604066248216, 0.002052578889661927],
                     [-0.41821536497073863, 0.24444849680948733, -0.8748376083079499],
                     [0.7531362119594922, -0.445118973113233, -0.48441195898731454]])
    Tvec = np.array([[2548.5812958074703], [-974.8825087558083], [5950.456044850678]])
    DIM = (1920, 1080)
    return K, D, Rvec, Tvec, DIM

