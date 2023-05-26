import time
import os


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
    print(intlst)
    for i in intlst:
        if os.path.exists(f"{directory}/{i}.jpg"):
            os.remove(f"{directory}/{i}.jpg")


current_path = os.getcwd()
while True:
    clean_history(300, current_path)
    time.sleep(300)