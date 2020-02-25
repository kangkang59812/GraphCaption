from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts4 as opts
import models
from dataloader4test import *
import eval_utils4 as eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='//home/lkk/code/self-critical.pytorch/log/log_aoanet2/model-234000.pth',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='//home/lkk/code/self-critical.pytorch/log/log_aoanet2/infos_aoanet2-234000.pkl',
                    help='path to infos to evaluate')


opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir',
           'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if k not in vars(opt):
            # copy over options from model
            vars(opt).update({k: vars(infos['opt'])[k]})

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance

loader = DataLoader(opt)

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                            vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis6.json', 'w'))
