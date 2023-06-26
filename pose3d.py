import time
from rootnetsum import findrootdepth, load_rootnet_data
from posenetsum import human_3d_detection, load_posenet_data
from yolo_opencv import get_bbox, load_yolo
import timeit
import paho.mqtt.client as mqtt
import os
import cv2
import psycopg2
from my_utils import gettime
from config import get_cam_parameter
from argparse import ArgumentParser
from config import DT_MQTT_HOST, DT_MQTT_TOPIC, DT_MQTT_USERNAME, DT_MQTT_PASSWORD, DT_MQTT_KEYFILEPATH, DT_MQTT_PORT, ORIGIN_WGS84, REFERENCE_LOCAL, REFERENCE_WGS84, DT_SQL, DT_SQL_LOCATION, DT_SQLCONNECTION
import numpy as np
from utm import project


def combined(image, net, classes, rootnet, posenet):
    bbox_list = get_bbox(image, classes, net)
    focal = [args.focal, args.focal]
    depth_list = findrootdepth(bbox_list, focal, image, rootnet)
    pload = human_3d_detection(bbox_list, depth_list, focal, image, posenet, Rvec, Tvec, angle, origin)
    return pload


def pose3d(net, classes, rootnet, posenet):
    lastimg = 0
    while True:
        t = gettime() - 30
        if lastimg != t:
            img_path = f"frames/{t}.jpg"
            if os.path.exists(img_path):
                start = timeit.default_timer()
                conn = psycopg2.connect(DT_SQLCONNECTION)
                cursor = conn.cursor()
                image = cv2.imread(img_path)
                pload = combined(image, net, classes, rootnet, posenet)
                sqldata = (t/10, DT_SQL_LOCATION, pload)
                client.publish(topicstr, payload=pload, qos=0, retain=False)
                cursor.execute(DT_SQL, sqldata)
                conn.commit()
                cursor.close()
                stop = timeit.default_timer()
                print(f"skeleton {t} published in {stop - start} second")
                lastimg = t
            else:
                print(f'waiting for image {t}')
                time.sleep(1)
        else:
            time.sleep(0.1)


def find_angle(c1, c2, origin):
    """
    :param:
    c1: projected x, y coordinates of WGS reference point
    c2: local reference x, y coordinates
    origin: projected x, y coordinates of WGS origin point
    :return: angle between 2 vectors from 2 coordinates respect to the origin in radians
    """
    vector_1 = np.subtract(c1, origin)
    vector_2 = c2
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--focal', type=int, default=1200, help='focal length')
    args = parser.parse_args()

    host = DT_MQTT_HOST

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("connected OK Returned code=", rc)
        else:
            print("Bad connection Returned code=", rc)


    # callback triggered by a new Pozyx data packet
    def on_publish(client, userdata, msg):
        print('msg published')


    # set client
    client = mqtt.Client(client_id='3dpose')
    # set callbacks
    client.on_connect = on_connect
    client.on_publish = on_publish
    # set credentials
    client.username_pw_set(DT_MQTT_USERNAME, password=DT_MQTT_PASSWORD)
    # set certificate
    client.tls_set(DT_MQTT_KEYFILEPATH)
    client.connect(host, port=DT_MQTT_PORT)
    client.loop_start()
    #topic
    topicstr = DT_MQTT_TOPIC

    # load trained model
    rootnet_model = load_rootnet_data()
    posenet_model = load_posenet_data()
    classes, net = load_yolo()

    # load camera parameters
    K, D, Rvec, Tvec, DIM = get_cam_parameter()

    # find angle between 2 local coordinate system
    origin = project(ORIGIN_WGS84)
    v2 = project(REFERENCE_WGS84)
    angle = find_angle((v2[2], v2[3]), REFERENCE_LOCAL, (origin[2], origin[3]))
    # start loop
    pose3d(net, classes, rootnet_model, posenet_model)



