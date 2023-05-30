# By Tonglong Song 2022/6/8 17:57
from scp import SCPClient
from my_utils import gettime, dt_connect
import os
import timeit
import time
from PIL import Image
from config import DT_IMGPATH


# posting image that is saved by frame_saver to dt cloud server
def postimg():
    c = dt_connect()
    lastpost = 0
    while True:
        t = gettime()-10
        if lastpost != t:
            if c.get_transport().is_alive():
                fname = f"3dpose/frames/{t}.jpg"
                if os.path.exists(fname):
                    try:
                        start = timeit.default_timer()
                        image = Image.open(fname)
                        image.save(fname, quality=70, optimize=True)
                        with SCPClient(c.get_transport()) as scp:
                            scp.put(f"3dpose/frames/{t}.jpg", DT_IMGPATH)
                        stop = timeit.default_timer()
                        print(f"{t}.jpg posted in {stop - start} second")
                        lastpost = t
                    except Exception as e:
                        print(e)
                else:
                    print('no image detected, check camera')
                    time.sleep(1)

            else:
                print("disconnected, trying to reconnect")
                try:
                    c = dt_connect()
                except Exception as e:
                    print(e)
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    time.sleep(5)
    postimg()