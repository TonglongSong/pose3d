import numpy as np

DT_HOSTNAME = "115.146.94.74"
DT_USERNAME = "ubuntu"
DT_KEYFILEPATH = "app1server.pem"
DT_IMGPATH = '/home/ubuntu/posecapture/tempdata/DLAB/'

DT_MQTT_HOST = 'mqtt.digitwin.com.au'
DT_MQTT_USERNAME = 'csdila-3dpose-publisher'
DT_MQTT_PASSWORD = 'ux693dD3x'
DT_MQTT_KEYFILEPATH = 'isrgrootx1.pem'
DT_MQTT_TOPIC = 'uom/parkville/melbourneconnect/level6/csdila/3dpose/humanSkeleton'
DT_MQTT_PORT = 8883

DT_SQLCONNECTION = "postgres://postgres:dtDBAdmin2021@45.113.235.78:5432/realtime_db"
DT_SQL = """INSERT INTO pose3d_history(epoch_time, location, json)
             VALUES(%s, %s, %s);"""
DT_SQL_LOCATION = "D_LAB"

ORIGIN_WGS84 = (144.964506, -37.800165)
REFERENCE_WGS84 = (144.964444, -37.800127)
REFERENCE_LOCAL = (0, 7)


def get_cam_parameter():
    K = np.array([[611.1549814728781, 0.0, 959.5], [0.0, 611.1549814728781, 539.5], [0.0, 0.0, 1.0]])
    D = np.array([[0.0], [0.0], [0.0], [0.0]])
    Rvec = np.array([[-0.5360341237889787, -0.8440653154164093, 0.014871497731556382],
                     [-0.24532120093734894, 0.13888997427130634, -0.9594410265449218],
                     [0.807765390756771, -0.517941423674448, -0.2815168825091088]])
    Tvec = np.array([[2399.768884141257], [245.79184214661134], [6078.2605312897995]])
    DIM = (1920, 1080)
    return K, D, Rvec, Tvec, DIM

