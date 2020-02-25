from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from functools import reduce
import torchvision
from collections import OrderedDict
import argparse
from dataloaderp import *
from tqdm import tqdm


class Head(nn.Module):
    '''
    use res101's structure before block2
    '''

    def __init__(self, model='res101', freeze=True):

        super(Head, self).__init__()
        if model == 'res101':
            base_model = torchvision.models.resnet101(
                pretrained=True)
            self.base_model = torch.nn.Sequential(OrderedDict([
                ('conv1', base_model.conv1),
                ('bn1', base_model.bn1),
                ('relu', base_model.relu),
                ('maxpool', base_model.maxpool),
                ('layer1', base_model.layer1),
            ]))

    def forward(self, x):

        out = self.base_model(x)

        return out


class Extract(nn.Module):
    def __init__(self, encoded_image_size=14, K=20, L=1024):
        super(Extract, self).__init__()

        self.head = Head(model='res101')

        model = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101
        self.features_model = torch.nn.Sequential(OrderedDict([
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4)
        ]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))
        # self.fine_tune2()
        # self.my_resnet.eval()
        del model
        torch.cuda.empty_cache()
        model = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101
        self.miml_intermidate = torch.nn.Sequential(OrderedDict([
            ('layer2', model.layer2),
            ('layer3', model.layer3)]))

        self.miml_last = torch.nn.Sequential(OrderedDict([
            ('layer4', model.layer4)]))

        dim = 2048
        map_size = 64
        self.K = K
        self.L = L
        self.miml_sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(dim, 512, 1)),
            ('dropout1', nn.Dropout(0.5)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, map_size)))
        ]))
        del model
        torch.cuda.empty_cache()
        self.freeze()
        self.freeze2()

    def freeze(self):
        for p in self.miml_intermidate.parameters():
            p.requires_grad = False
        for p in self.miml_last.parameters():
            p.requires_grad = False
        for p in self.miml_sub_concept_layer.parameters():
            p.requires_grad = False

    def freeze2(self):
        for p in self.features_model.parameters():
            p.requires_grad = False

    def forward(self, images):

        # Prepare the features
        batch_size = images.shape[0]
        head_out = self.head(images)
        # miml
        miml_features_out = self.miml_last(self.miml_intermidate(head_out))
        # (-1,2048,8,8)
        _, C, H, W = miml_features_out.shape
        conv1_out = self.miml_sub_concept_layer.dropout1(
            self.miml_sub_concept_layer.conv1(miml_features_out))
        # shape -> (n_bags, (L * K), n_instances, 1)
        conv2_out = self.miml_sub_concept_layer.conv2(conv1_out)
        # shape -> (n_bags, L, K, n_instances)
        conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
        # shape -> (n_bags, L, 1, n_instances),remove dim: 1
        maxpool1_out = self.miml_sub_concept_layer.maxpool1(
            conv2_out).squeeze(2)
        # softmax
        permute1 = maxpool1_out.permute(0, 2, 1)
        softmax1 = self.miml_sub_concept_layer.softmax1(permute1)
        permute2 = softmax1.permute(0, 2, 1)
        # reshape = permute2.unsqueeze(2)
        # predictions_instancelevel
        reshape = permute2.reshape(-1, self.L, 1, H*W)
        # shape -> (n_bags, L, 1, 1)
        maxpool2_out = self.miml_sub_concept_layer.maxpool2(reshape)
        attributes = maxpool2_out.squeeze()

        # extract image features
        # (batch_size, 2048, image_size/32, image_size/32)
        images_features = self.features_model(head_out)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        imgs_features = self.adaptive_pool(images_features)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        imgs_features = imgs_features.permute(
            0, 2, 3, 1).reshape(batch_size, -1, 2048)

        return attributes, imgs_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='/home/lkk/code/self-critical.pytorch/data/cocotalk.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--num_worker', type=int, default=0,
                        help='')
    parser.add_argument('--input_sg_dir', type=str, default='/home/lkk/code/self-critical.pytorch/data/coco_img_sg',
                        help='scene graph')
    parser.add_argument('--input_sg_voc', type=str, default='/home/lkk/code/self-critical.pytorch/data/coco_pred_sg_rela.npy',
                        help='scene graph voc')
    parser.add_argument('--input_label_h5', type=str,
                        default='/home/lkk/code/self-critical.pytorch/data/cocotalk_label.h5')
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')
    opt = parser.parse_args()
    loader = DataLoaderRaw(opt)
    model = Extract().cuda()
    model.eval()
    for i in tqdm(range(7081)):
        data = loader.get_batch('train')
        attrs, imgs = model(data['img'].cuda())
        attrs, imgs = attrs.detach().cpu().numpy(), imgs.detach().cpu().numpy()
        infos = data['infos']
        for j in range(len(infos)):
            name = str(infos[j]['id'])
            attr = attrs[j]
            img = imgs[j]
            path = os.path.join(
                '/home/lkk/code/self-critical.pytorch/data/cocoresnet', name)
            np.savez(path, attr=attr, img=img)
    
    for i in tqdm(range(313)):
        data = loader.get_batch('val')
        attrs, imgs = model(data['img'].cuda())
        attrs, imgs = attrs.detach().cpu().numpy(), imgs.detach().cpu().numpy()
        infos = data['infos']
        for j in range(len(infos)):
            name = str(infos[j]['id'])
            attr = attrs[j]
            img = imgs[j]
            path = os.path.join(
                '/home/lkk/code/self-critical.pytorch/data/cocoresnet', name)
            np.savez(path, attr=attr, img=img)
    
    for i in tqdm(range(313)):
        data = loader.get_batch('test')
        attrs, imgs = model(data['img'].cuda())
        attrs, imgs = attrs.detach().cpu().numpy(), imgs.detach().cpu().numpy()
        infos = data['infos']
        for j in range(len(infos)):
            name = str(infos[j]['id'])
            attr = attrs[j]
            img = imgs[j]
            path = os.path.join(
                '/home/lkk/code/self-critical.pytorch/data/cocoresnet', name)
            np.savez(path, attr=attr, img=img)
