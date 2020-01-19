import numpy as np
import os
import pickle
import math
from tqdm import tqdm

yichang = np.array([[425.41144,   0., 639.2, 479.2],
                    [0., 209.01892, 446.58472, 479.2],
                    [37.065624,   0., 467.56372, 403.38495],
                    [31.646069, 181.52281, 639.2, 479.2],
                    [291.29047,   0., 556.3114, 479.2],
                    [4.838501,   0., 639.2, 218.58691],
                    [96.25258,   0., 432.91107, 479.2],
                    [482.9079,  10.244525, 583.4144, 118.63218],
                    [456.39322, 296.8479, 623.72156, 472.1024],
                    [0., 329.16315, 561.1102, 479.2],
                    [36.726196, 222.28362, 263.2574, 468.6128],
                    [283.29263, 208.3899, 381.17, 314.19073],
                    [210.09421, 199.59758, 405.85324, 427.1874],
                    [242.75029, 187.77534, 462.7431, 463.34174],
                    [0., 345.92828, 249.85266, 479.2],
                    [432.5979, 134.41025, 622.3935, 342.30884],
                    [148.65598, 299.053, 366.59564, 479.2],
                    [148.61118, 373.4122, 305.32843, 457.97314],
                    [289.8145, 202.179, 384.8266, 291.73315],
                    [41.11455, 240.37146, 178.73335, 348.72162],
                    [0.,   2.596936,  94.506386, 203.10837],
                    [411.70273, 218.61258, 639.2, 479.2],
                    [493.73233,  55.904247, 544.82263, 100.789345],
                    [36.27781, 237.87163, 163.97392, 321.90616],
                    [427.5771,  68.469284, 524.05994, 155.67178],
                    [0.,   0.,  86.14436, 355.6267],
                    [3.20784,  45.891586,  60.37638, 122.31523],
                    [467.13202,   9.627314, 596.4172, 176.40234]])


def get_cwh(box):
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    return cx, cy, w, h


boxInfo = pickle.load(open('data/vsua_box_info.pkl', 'rb'), encoding='latin1')
dirr = '/home/lkk/code/self-critical.pytorch/data/coco_img_sg'
bdirr = '/home/lkk/code/self-critical.pytorch/data/cocobu_box'
names = os.listdir(dirr)
NumFeats = 8
for name in tqdm(names):
    path = os.path.join(dirr, name)
    bpath = os.path.join(bdirr, name)
    data = np.load(path, encoding='latin1').item()
    boxex = np.load(bpath)
    index = data['rela_matrix'].astype(np.int)[:, 0:2]
    num_rela = index.shape[0]
    # boxex = boxInfo[int(name.split('.')[0])]['boxes']
    h = boxInfo[int(name.split('.')[0])]['image_h']
    w = boxInfo[int(name.split('.')[0])]['image_w']
    scale = w * h
    diag_len = math.sqrt(w ** 2 + h ** 2)
    feats = np.zeros([num_rela, NumFeats], dtype='float')

    for i, pair in enumerate(index):

        box1, box2 = boxex[pair[0]], boxex[pair[1]]

        cx1, cy1, w1, h1 = get_cwh(box1)
        cx2, cy2, w2, h2 = get_cwh(box2)
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        # scale
        scale1 = w1 * h1
        scale2 = w2 * h2
        # Offset
        offsetx = cx2 - cx1
        offsety = cy2 - cy1
        # Aspect ratio
        aspect1 = w1 / h1
        aspect2 = w2 / h2
        # Overlap
        i_xmin = max(x_min1, x_min2)
        i_ymin = max(y_min1, y_min2)
        i_xmax = min(x_max1, x_max2)
        i_ymax = min(y_max1, y_max2)
        iw = max(i_xmax - i_xmin + 1, 0)
        ih = max(i_ymax - i_ymin + 1, 0)
        areaI = iw * ih
        areaU = scale1 + scale2 - areaI
        # dist
        dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        # angle
        angle = math.atan2(cy2 - cy1, cx2 - cx1)

        f1 = offsetx / math.sqrt(scale1)
        f2 = offsety / math.sqrt(scale1)
        f3 = math.sqrt(scale2 / scale1)
        f4 = areaI / areaU
        f5 = aspect1
        f6 = aspect2
        f7 = dist / diag_len
        f8 = angle
        feat = [f1, f2, f3, f4, f5, f6, f7, f8]
        feats[i] = np.array(feat)

    save_name = os.path.join(
        '/home/lkk/code/self-critical.pytorch/data/cocobu_geometry', name.split('.')[0])
    np.savez(save_name, feats=feats, image_h=h, image_w=w)
