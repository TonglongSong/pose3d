import time
import os
import paramiko
from config import DT_HOSTNAME, DT_USERNAME, DT_KEYFILEPATH


def dt_connect():
    k = paramiko.RSAKey.from_private_key_file(DT_KEYFILEPATH)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("connecting")
    c.connect(hostname=DT_HOSTNAME, username=DT_USERNAME, pkey=k)
    print("connected")
    return c


def gettime():
    t = time.time()
    diff = t - int(t)
    if diff >= 0.5:
        return int(t)*10+5
    else:
        return int(t)*10


def clean_history(t, directory):
    namelst = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    namelst = [i[:-4] for i in namelst]
    intlst = []
    for name in namelst:
        try:
            intname = int(name)
            intlst.append(intname)
        except Exception:
            pass
    intlst = [i for i in intlst if i < gettime()-t*10]
    for i in intlst:
        if os.path.exists(f"{directory}/{i}.jpg"):
            os.remove(f"{directory}/{i}.jpg")

