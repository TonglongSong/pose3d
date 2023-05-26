import numpy as np


def get_cam_parameter(cam_num):
    if cam_num == 0:
        K = np.array([[611.1549814728781, 0.0, 959.5], [0.0, 611.1549814728781, 539.5], [0.0, 0.0, 1.0]])
        D = np.array([[0.0], [0.0], [0.0], [0.0]])
        Rvec = np.array([[-0.5078205930618596, -0.8614604066248216, 0.002052578889661927],
                         [-0.41821536497073863, 0.24444849680948733, -0.8748376083079499],
                         [0.7531362119594922, -0.445118973113233, -0.48441195898731454]])
        Tvec = np.array([[2548.5812958074703], [-974.8825087558083], [5950.456044850678]])
        return K, D, Rvec, Tvec
    if cam_num == 1:
        K = np.array(
            [[940.0570267048228, 0.0, 981.6062514851349], [0.0, 939.1948373847227, 586.0875107741564], [0.0, 0.0, 1.0]])
        D = np.array([[-0.05389428814161064], [0.05590081942628492], [-0.06749300811309662], [0.025889007600741934]])
        Rvec = np.array([[-0.19627610561542297, 0.980417974663796, 0.016008913785253542],
                         [0.6780925930071715, 0.14750866130704077, -0.7200219650456612],
                         [-0.7082839301249562, -0.13046758139685594, -0.6937665922565031]])
        Tvec = np.array([[-2653.788179592122], [-2083.20020967356], [6057.821318953561]])
        return K, D, Rvec, Tvec
    if cam_num == 2:
        K = np.array([[986.4372705717541, 0.0, 898.8835980612882], [0.0, 979.2285354702622, 428.20450565999454],
                      [0.0, 0.0, 1.0]])
        D = np.array([[-0.008715934710700009], [-0.09835625257753927], [0.1484167382016628], [-0.08203574257186709]])
        Rvec = np.array([[-0.947871076226325, 0.31853588313877257, -0.008678364278509737],
                         [0.026794978340675235, 0.052537019721326406, -0.9982594305562673],
                         [-0.31752551391859996, -0.9464537773773227, -0.058333483514409146]])
        Tvec = np.array([[167.98607291825797], [178.95971387610052], [601.7620719305771]])
        return K, D, Rvec, Tvec
    if cam_num == 3:
        K = np.array([[931.2134308342625, 0.0, 1075.3379845552727], [0.0, 934.2184110874448, 560.2449102238267],
                      [0.0, 0.0, 1.0]])
        D = np.array([[-0.048329921844457086], [0.057868253127834494], [-0.07353578301108772], [0.032514425582209905]])
        Rvec = np.array([[-0.9143747272954343, -0.40283776587760556, 0.04050422775636492],
                         [-0.076280591494053, 0.07316020108038801, -0.994398741119072],
                         [0.3976180698166488, -0.9123427641849402, -0.0976245419690448]])
        Tvec = np.array([[129.23806528970945], [154.10852992314486], [567.0514278662093]])
        return K, D, Rvec, Tvec
    if cam_num == 4:
        K = np.array(
            [[941.8129295477133, 0.0, 851.2574389822521], [0.0, 940.5300869167609, 441.7784093232078], [0.0, 0.0, 1.0]])
        D = np.array([[-0.007068117469651826], [-0.10627639099230048], [0.1501849420018641], [-0.07287609783690653]])
        Rvec = np.array([[0.2537980764329387, -0.966609104614568, 0.035403040479078884],
                         [-0.08335494984106456, -0.058322115400248675, -0.9948117827972452],
                         [0.9636589068420078, 0.24953029822340855, -0.09537369413314667]])
        Tvec = np.array([[99.29718178076442], [163.69987666629243], [270.785303296209]])
        return K, D, Rvec, Tvec

