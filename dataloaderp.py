from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
import skimage
import skimage.io
import scipy.misc
import torch.utils.data as data
from torchvision import transforms as trn

import PIL
import argparse

preprocess = trn.Compose([
    trn.Resize((256, 256)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext

        if 'resnet' in db_path:
            self.loader = lambda x: (np.load(x, allow_pickle=True)['attr'],
                                     np.load(x, allow_pickle=True)['img'])

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


class DataLoaderRaw(data.Dataset):

    def __init__(self, opt):
        self.opt = opt

        self.nw = opt.num_worker
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        # self.sg_loader = HybridLoader(self.opt.input_sg_dir, '.npy')
        # Load resnet data
        self.data_loader = HybridLoader(self.opt.res_data, '.npz')
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        # load the json file which contains additional information about the dataset

        self.files = []
        self.ids = []

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

        self.num_images = len(self.info['images'])
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            fullpath = os.path.join(
                '/home/lkk/datasets/coco2014', img['file_path'])
            self.files.append(fullpath)
            self.ids.append(img['id'])
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

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        image_id = str(self.info['images'][ix]['id'])
        attr, img = self.data_loader.get(image_id)
        # img = skimage.io.imread(self.files[self.ids.index(int(image_id))])
        # if len(img.shape) == 2:
        #     img = img[:, :, np.newaxis]
        #     img = np.concatenate((img, img, img), axis=2)
        # # img = img[:,:,:3].astype('float32')/255.0
        # tmp_img = preprocess(PIL.Image.fromarray(img))

        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)

        return attr, img, seq, ix

    def __len__(self):
        return len(self.info['images'])

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = self.seq_per_img

        # pick an index of the datapoint to load next

        infos = []
        # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        attr_batch = []
        img_batch = []
        # np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        label_batch = []
        wrapped = False
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_attr, tmp_img, tmp_seq, ix, tmp_wrapped = self._prefetch_process[split].get(
            )
            if tmp_wrapped:
                wrapped = True

            img_batch.append(tmp_img)
            attr_batch.append(tmp_attr)
            tmp_label = np.zeros(
                [seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            if hasattr(self, 'h5_label_file'):
                gts.append(
                    self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get(
                'file_path', '')
            infos.append(info_dict)

        data = {}
        data['attr'] = np.vstack([_]*seq_per_img for _ in attr_batch)
        data['img'] = np.vstack([_]*seq_per_img for _ in img_batch)

        data['labels'] = np.vstack(label_batch)
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

        data = {k: torch.from_numpy(v) if type(
            v) is np.ndarray else v for k, v in data.items()}

        return data

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
                        default=8)
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')
    parser.add_argument('--res_data', type=str, default='/home/lkk/code/self-critical.pytorch/data/cocoresnet',
                        help='path to the res data containing the preprocessed dataset')
    parser.add_argument('--seq_per_img', type=int, default=5,
                        help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    opt = parser.parse_args()
    loader = DataLoaderRaw(opt)
    data = loader.get_batch('train')
    print(data)
