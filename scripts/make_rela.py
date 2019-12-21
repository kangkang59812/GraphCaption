import numpy as np
import os
from tqdm import tqdm
dirr = '/home/lkk/code/self-critical.pytorch/data/coco_img_sg'
names = os.listdir(dirr)


# sg_data =
#         objs_label = sg_data['obj_attr'][:, 1]
#         rela_matrix = sg_data['rela_matrix']
for name in tqdm(names):
    path = os.path.join(dirr, name)
    data = np.load(path, encoding='latin1').item()
    index = data['rela_matrix'].astype(np.int)[:, 0:2]
    num_obj = data['obj_attr'].shape[0]
    num_rela = data['rela_matrix'].shape[0]

    rela_adj = np.zeros((num_obj, num_rela))
