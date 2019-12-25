from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import lmdb
import os
import numpy as np
import random
import torch
import torch.utils.data as data
import six
import pdb

# 无关系
exclude = [481399, 110026, 317035, 563175, 516124, 317431, 510418, 514772, 88173, 84548, 224733,
           37157, 447337, 496065, 515062, 560360, 163361, 83730, 76138, 423141, 406531, 46422, 295626,
           43347, 322211, 222990, 350067, 391689, 180515, 504382, 156002, 453348, 90365, 119718]


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x, allow_pickle=True)
        else:
            self.loader = lambda x: np.load(x, allow_pickle=True)['feat']
        if 'sg' in db_path:
            self.loader = lambda x: np.load(
                x, encoding='latin1', allow_pickle=True).item()
        if 'adj' in db_path:
            # 必须是对象，不可以是file
            self.loader = lambda x: (np.load(x, allow_pickle=True)['adj1'], np.load(x)[
                                     'adj2'], np.load(x)['adj3'])
        if 'geometry' in db_path:
            self.loader = lambda x: np.load(x, allow_pickle=True)['feats']

        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)
        elif db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        else:
            self.db_type = 'dir'

    def get(self, key):

        if self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key)
            f_input = six.BytesIO(byteflow)
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        else:
            f_input = os.path.join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)

        return feat


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(
            split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.nw = opt.num_worker
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)

        self.sg_voc = np.load(self.opt.input_sg_voc,
                              allow_pickle=True).item()['rela_dict']

        # open the hdf5 file
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(
                self.opt.input_label_h5, 'r')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy')
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz')
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy')
        self.sg_loader = HybridLoader(self.opt.input_sg_dir, '.npy')
        self.adj_loader = HybridLoader(self.opt.input_adj, '.npz')
        self.geometry_loader = HybridLoader(self.opt.geometry_dir, '.npz')

        # self.label_start_ix.shape[0]
        self.num_images = len(self.info['images'])
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if 'split' not in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %
              len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(
                split, self, split == 'train')
        # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = self.seq_per_img

        # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        fc_batch = []
        # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        att_batch = []
        # np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        label_batch = []
        adj1_batch = []
        adj2_batch = []
        adj3_batch = []
        geometry = []
        rela_label = []
        obj_label = []
        sub_obj = []
        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_seq, tmp_adj, tmp_geometry, tmp_rela_label, tmp_obj_label, tmp_sub_obj, \
                ix, tmp_wrapped = self._prefetch_process[split].get()
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            adj1_batch.append(tmp_adj[0])  # adj1
            adj2_batch.append(tmp_adj[1])  # adj2
            adj3_batch.append(tmp_adj[2])  # adj3
            geometry.append(tmp_geometry)
            rela_label.append(tmp_rela_label)
            obj_label.append(tmp_obj_label)
            sub_obj.append(tmp_sub_obj)
            tmp_label = np.zeros(
                [seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                gts.append(
                    self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get(
                'file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        fc_batch, att_batch, label_batch, gts, infos, adj1, adj2, adj3, geometry, \
            rela_label, obj_label, sub_obj = zip(*sorted(zip(fc_batch, att_batch, label_batch,
                                                             gts, infos, adj1_batch, adj2_batch, adj3_batch, geometry, rela_label, obj_label, sub_obj), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(
            sum([[_]*seq_per_img for _ in fc_batch], []))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i *
                              seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(
            data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i *
                              seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask, 包括开头和结束标志和正文，不包括补位的0
        nonzeros = np.array(
            list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        mask_batch = np.zeros(
            [data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        max_rela_len = max([_.shape[0] for _ in rela_label])
        # 计算边特征，用到的节点特征的邻接矩阵
        data['rela_sub'] = np.zeros(
            [len(att_batch)*seq_per_img, max_rela_len, max_att_len], dtype='float32')
        data['rela_obj'] = np.zeros(
            [len(att_batch)*seq_per_img, max_rela_len, max_att_len], dtype='float32')
        for i in range(len(att_batch)):
            tmp_rela_sub = np.zeros([max_rela_len, max_att_len], dtype='float32')
            tmp_rela_obj = np.zeros([max_rela_len, max_att_len], dtype='float32')

            for j, item in enumerate(sub_obj[i]):
                tmp_rela_sub[j, item[0]] = 1
                tmp_rela_obj[j, item[1]] = 1
            # 有些图没有节点关系
            # if adj1[i].size == 0:
            #     tmp_rela_adj = np.zeros(
            #         [max_att_len, max_rela_len], dtype='int')
            tmp_rela_sub = tmp_rela_sub[np.newaxis, :]
            tmp_rela_obj = tmp_rela_obj[np.newaxis, :]

            tmp_rela_sub = np.repeat(tmp_rela_sub, seq_per_img, axis=0)
            tmp_rela_obj = np.repeat(tmp_rela_obj, seq_per_img, axis=0)

            data['rela_sub'][i*seq_per_img:(i+1)*seq_per_img, :] = tmp_rela_sub
            data['rela_obj'][i*seq_per_img:(i+1)*seq_per_img, :] = tmp_rela_obj

        data['rela_label'] = np.zeros(
            [len(att_batch)*seq_per_img, max_rela_len], dtype='int')
        for i in range(len(att_batch)):
            data['rela_label'][i *
                               seq_per_img:(i+1)*seq_per_img, :rela_label[i].shape[0]] = rela_label[i]
            if rela_label[i].size == 0:
                data['rela_label'][i *
                                   seq_per_img:(i+1)*seq_per_img, 0] = 0

        data['rela_masks'] = np.zeros(
            data['rela_label'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['rela_masks'][i *
                               seq_per_img:(i+1)*seq_per_img, :adj1[i].shape[0]] = 1
            if adj1[i].size == 0:
                data['rela_masks'][i *
                                   seq_per_img:(i+1)*seq_per_img, 0] = 1

        # 计算节点特征，用到的边的特征的邻接矩阵
        data['rela_n2r'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len, max_rela_len], dtype='float32')
        for i in range(len(att_batch)):

            tmp_rela_adj = np.zeros([max_att_len, max_rela_len], dtype='float32')
            for j, item in enumerate(adj1[i]):
                tmp_rela_adj[item[0], j] = 1.
            # 有些图没有节点关系
            # if adj1[i].size == 0:
            #     tmp_rela_adj = np.zeros(
            #         [max_att_len, max_rela_len], dtype='int')
            tmp_rela_adj = tmp_rela_adj[np.newaxis, :]
            tmp_rela_adj = np.repeat(tmp_rela_adj, seq_per_img, axis=0)
            data['rela_n2r'][i*seq_per_img:(i+1)*seq_per_img, :] = tmp_rela_adj

        data['obj_label'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len], dtype='int')
        for i in range(len(att_batch)):
            data['obj_label'][i *
                              seq_per_img:(i+1)*seq_per_img, :obj_label[i].shape[0]] = obj_label[i]

        data['geometry'] = np.zeros(
            [len(att_batch)*seq_per_img, max_rela_len, 8], dtype='float32')
        for i in range(len(att_batch)):
            data['geometry'][i *
                             seq_per_img:(i+1)*seq_per_img, :adj1[i].shape[0]] = geometry[i]

        data['adj1'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len, max_att_len], dtype='float32')
        data['adj2'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len, max_att_len], dtype='float32')
        data['adj3'] = np.zeros(
            [len(att_batch)*seq_per_img, max_att_len, max_att_len], dtype='float32')

        for i in range(len(att_batch)):
            adj_dis1 = np.zeros(
                (max_att_len, max_att_len))
            adj_dis2 = np.zeros(
                (max_att_len, max_att_len))
            adj_dis3 = np.zeros(
                (max_att_len, max_att_len))

            if adj1_batch[i].shape[0] != 0:
                adj_dis1[adj1_batch[i][:, 0], adj1_batch[i][:, 1]] = 1
            adj_dis1 = adj_dis1[np.newaxis, :]
            adj_dis1 = np.repeat(adj_dis1, seq_per_img, axis=0)
            data['adj1'][i * seq_per_img:(i+1)*seq_per_img, :] = adj_dis1

            if adj2_batch[i].shape[0] != 0:
                adj_dis2[adj2_batch[i][:, 0], adj2_batch[i][:, 1]] = 1
            adj_dis2 = adj_dis2[np.newaxis, :]
            adj_dis2 = np.repeat(adj_dis2, seq_per_img, axis=0)
            data['adj2'][i * seq_per_img:(i+1)*seq_per_img, :] = adj_dis2

            if adj3_batch[i].shape[0] != 0:
                adj_dis3[adj3_batch[i][:, 0], adj3_batch[i][:, 1]] = 1
            adj_dis3 = adj_dis3[np.newaxis, :]
            adj_dis3 = np.repeat(adj_dis3, seq_per_img, axis=0)
            data['adj3'][i * seq_per_img:(i+1)*seq_per_img, :] = adj_dis3

        # Turn all ndarray to torch tensor
        data = {k: torch.from_numpy(v) if type(
            v) is np.ndarray else v for k, v in data.items()}

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]

        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / \
                    np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(
                    str(self.info['images'][ix]['id']))
                # devided by image width and height
                x1, y1, x2, y2 = np.hsplit(box_feat, 4)
                h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                # question? x2-x1+1??
                box_feat = np.hstack(
                    (x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h)))
                if self.norm_box_feat:
                    box_feat = box_feat / \
                        np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(
                    sorted(att_feat, key=lambda x: x[-1], reverse=True))
        else:
            att_feat = np.zeros((1, 1, 1), dtype='float32')
        if self.use_fc:
            fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
        else:
            fc_feat = np.zeros((1), dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        sg_data = self.sg_loader.get(str(self.info['images'][ix]['id']))

        adj = self.adj_loader.get(str(self.info['images'][ix]['id']))

        geometry = self.geometry_loader.get(str(self.info['images'][ix]['id']))

        rela_label = sg_data['rela_matrix'][:, 2].astype(np.int)-408+1
        obj_label = sg_data['obj_attr'][:, 1].astype(np.int)+1
        sub_obj = sg_data['rela_matrix'][:, 0:2].astype(np.int)
        return (fc_feat,
                att_feat, seq, adj, geometry, rela_label, obj_label, sub_obj,
                ix)

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(
                                                     self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.dataloader.nw,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        # while self.dataloader.info['images'][ix]['id'] in exclude:
        #     ix, wrapped = self._get_next_minibatch_inds()
        #     tmp = self.split_loader.next()

        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]
