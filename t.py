from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import misc.resnet
from misc.resnet_utils import myResnet
import json
import h5py
import os
import numpy as np
import random
import torch
import skimage
import skimage.io
import scipy.misc

from torchvision import transforms as trn

cnn_model = 'resnet101'
my_resnet = getattr(misc.resnet, cnn_model)()
my_resnet.load_state_dict(torch.load(
    "/home/lkk/.torch/models/resnet101-5d3b4d8f.pth"))
my_resnet = myResnet(my_resnet)
my_resnet.cuda()
my_resnet.eval()

