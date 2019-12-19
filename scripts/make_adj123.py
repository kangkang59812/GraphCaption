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

    # 一阶邻居
    Adj_dis1 = np.zeros((num_obj, num_obj))
    Adj_dis1[index[:, 0], index[:, 1]] = 1
    assert np.sum(Adj_dis1) == index.shape[0]
    # print('一阶邻居')
    # print(index)
    # print('**********************')

    graph = dict()
    for item in index:
        if item[0] not in graph:
            graph[item[0]] = list()
            graph[item[0]].append(item[1])
        else:
            graph[item[0]].append(item[1])
    adj2 = list()
    for k, v in graph.items():
        for one in v:
            if one in graph.keys():
                for i in graph[one]:
                    adj2.append([k, i])
    # print('二阶邻居')
    # print(len(adj2))
    # print(adj2)
    adj2 = np.array(adj2)
    Adj_dis2 = np.zeros((num_obj, num_obj))
    if adj2.shape[0] != 0:
        Adj_dis2[adj2[:, 0], adj2[:, 1]] = 1
    # print('sumAdj_dis2--{}'.format(np.sum(Adj_dis2)))
    # print(Adj_dis2)
    # print('**********************')

    graph2 = dict()
    for item in adj2:
        if item[0] not in graph2:
            graph2[item[0]] = list()
            graph2[item[0]].append(item[1])
        else:
            graph2[item[0]].append(item[1])

    adj3 = list()
    for k, v in graph2.items():
        for one in v:
            if one in graph.keys():
                for i in graph[one]:
                    adj3.append([k, i])
    # print('三阶邻居')
    # print(len(adj3))
    # print(adj3)
    adj3 = np.array(adj3)
    Adj_dis3 = np.zeros((num_obj, num_obj))
    if adj3.shape[0] != 0:
        Adj_dis3[adj3[:, 0], adj3[:, 1]] = 1
    # print('sumAdj_dis3--{}'.format(np.sum(Adj_dis3)))
    # print(Adj_dis3)
    # print(np.nonzero(Adj_dis3))

    save_name = os.path.join(
        '/home/lkk/code/self-critical.pytorch/data/coco_img_sg_adj', name.split('.')[0])
    np.savez(save_name,
             adj1=index, adj2=adj2, adj3=adj3)

    # # 二阶邻居
    # Adj_dis2 = np.zeros((num_obj, num_obj))
    # n1 = index[:, 0]
    # n2 = index[:, 1]
    # # n2中元素是否在n1中，结果形状和n2相同
    # mask = np.isin(n2, n1)
    # # n1中对应的元素位置
    # ins = list(set(n2[mask]))
    # mask2 = np.isin(n1, ins)

    # d = dict()
    # for i in ins:
    #     d[i] = list()

    # for one in ins:
    #     for item in index:
    #         if one == item[0]:
    #             d[one].append(item[1])

    # result = list()

    # for item in index[mask, :]:
    #     for i in d[item[1]]:
    #         result.append([item[0], i])

    # result = np.array(result)
    # Adj_dis2[result[:, 0], result[:, 1]] = 1
    # print(len(result))
    # print(result)

    # Adj_dis3 = np.zeros((num_obj, num_obj))
    # mask3 = np.isin(n1, result[:, 1])
    # ins2 = list(set(n1[mask3]))
    # # print(ins2)
    # d2 = dict()
    # for i in ins2:
    #     d2[i] = list()

    # for i in ins2:
    #     for item in index:
    #         if i == item[0]:
    #             d2[i].append(item[1])
    # # print(d2)
    # result2 = list()
    # for item in result:
    #     if item[1] in d2.keys():
    #         for i in d2[item[1]]:
    #             result2.append([item[0], i])
    # print('*****************')
    # print(len(result2))
    # print(result2)
