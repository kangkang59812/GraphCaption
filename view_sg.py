import numpy as np
import json

data = 'trainval'  # trainval or test
image_id = '109'  # 14, 16

if data == 'trainval':
    sg_path = '/home/lkk/code/self-critical.pytorch/data/coco_img_sg/'+image_id+'.npy'
    feat_path = '/home/lkk/code/self-critical.pytorch/data/cocobu_att/'+image_id+'.npz'
    voc = np.load(
        '/home/lkk/code/self-critical.pytorch/data/coco_pred_sg_rela.npy').item()['rela_dict']
    box_path = '/home/lkk/code/self-critical.pytorch/data/cocobu_box/'+image_id+'.npy'
    
    sg = np.load(sg_path, encoding='latin1').item()
    feat = np.load(feat_path)
    box = np.load(box_path)

    print('sg object numbers:{}'.format(sg['obj_attr'].shape[0]))
    print('object numbers:{}'.format(feat['feat'].shape[0]))

    assert sg['obj_attr'].shape[0] == feat['feat'].shape[0]
    rela_matrix = sg['rela_matrix']
    obj_attr = sg['obj_attr']

    for i in range(rela_matrix.shape[0]):
        sub_index = int(rela_matrix[i][0])
        obj_index = int(rela_matrix[i][1])
        sub = obj_attr[sub_index][1]
        obj = obj_attr[obj_index][1]

        sub_name = voc[sub]
        obj_name = voc[obj]

        rela = int(rela_matrix[i][2])
        rela_name = voc[rela]
        print('{}---{}---{}'.format(sub_name, rela_name, obj_name))
        print('')
elif data == 'test':
    sg_path = '/home/lkk/code/self-critical.pytorch/coco_pred_sg_test.npy'
    feat_path = '/home/lkk/code/self-critical.pytorch/data/testcocobu_att/'+image_id+'.npz'
    voc = np.load(
        '/home/lkk/code/self-critical.pytorch/data/coco_pred_sg_rela.npy').item()['rela_dict']

    sg = np.load(sg_path, encoding="latin1").item()[str(image_id)]
    feat = np.load(feat_path)

    print('sg object numbers:{}'.format(sg['obj_pred'].shape[0]))
    print('object numbers:{}'.format(feat['feat'].shape[0]))

    assert sg['obj_pred'].shape[0] == feat['feat'].shape[0]

    rela_pred = sg['rela_pred']
    rela_matrix = sg['rela_matrix']
    obj_attr = sg['obj_pred']

    for i in range(rela_matrix.shape[0]):
        sub_index = int(rela_matrix[i][0])
        obj_index = int(rela_matrix[i][1])
        sub = obj_attr[sub_index][1]
        obj = obj_attr[obj_index][1]

        sub_name = voc[sub]
        obj_name = voc[obj]

        rela = rela_pred[i][0]
        rela_name = voc[rela+408]
        print('{}---{}---{}'.format(sub_name, rela_name, obj_name))

else:
    pass
