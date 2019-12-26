from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import misc.utils as utils
from .CaptionModel import CaptionModel

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with',
               'before', 'after', 'on', 'upon', 'near', 'to', 'is', 'are', 'am']
bad_endings += ['the']


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(
        input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(
            att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        # 用作节点特征
        self.h2node = nn.Linear(self.rnn_size, self.att_hid_size)
        # 用作关系特征
        self.h2rela = nn.Linear(self.rnn_size, self.att_hid_size)

        self.alpha_net1 = nn.Linear(self.att_hid_size, 1)
        self.alpha_net2 = nn.Linear(self.att_hid_size, 1)

        self.node_gate = nn.Sequential(nn.Dropout(0.5), nn.Linear(
            2*self.att_hid_size, 2*self.att_hid_size), nn.GLU())
        self.rela_gate = nn.Sequential(nn.Dropout(0.5), nn.Linear(
            2*self.att_hid_size, 2*self.att_hid_size), nn.GLU())

    def forward(self, h, node_feats, p_node_feats, rela_feats, p_rela_feats, att_masks=None, rela_masks=None):
        node_size = node_feats.numel() // node_feats.size(0) // node_feats.size(-1)
        node = p_node_feats.view(-1, node_size, self.att_hid_size)

        rela_size = rela_feats.numel() // rela_feats.size(0) // rela_feats.size(-1)
        rela = p_rela_feats.view(-1, rela_size, self.att_hid_size)

        # The p_att_feats here is already projected, 单卡这样
        # node_size = node_feats.shape[1]
        # rela_size = rela_feats.shape[1]

        node_h = self.h2node(h)
        rela_h = self.h2rela(h)
        # [1280,512]-->[1280,61,512]; [1280,512]-->1280,30,512]
        node_h = node_h.unsqueeze(1).expand_as(node_feats)
        rela_h = rela_h.unsqueeze(1).expand_as(rela_feats)
        # batch * att_size * att_hid_size
        node_dot = p_node_feats + node_h
        rela_dot = p_rela_feats + rela_h

        # batch * att_size * att_hid_size
        node_dot = torch.tanh(node_dot)
        rela_dot = torch.tanh(rela_dot)

        # (batch * att_size) * att_hid_size
        node_dot = node_dot.view(-1, self.att_hid_size)
        rela_dot = rela_dot.view(-1, self.att_hid_size)
        # (batch * att_size) * 1
        node_dot = self.alpha_net1(node_dot)
        rela_dot = self.alpha_net2(rela_dot)

        # [1280,61]; [1280,30]
        node_dot = node_dot.view(-1, node_size)
        rela_dot = rela_dot.view(-1, rela_size)

        # batch * att_size
        node_weight = F.softmax(node_dot, dim=1)
        rela_weight = F.softmax(rela_dot, dim=1)

        if att_masks is not None:
            node_weight = node_weight * att_masks.view(-1, node_size).float()
            node_weight = node_weight / \
                node_weight.sum(1, keepdim=True)  # normalize to 1
        if rela_masks is not None:
            rela_weight = rela_weight * rela_masks.view(-1, rela_size).float()
            rela_weight = rela_weight / \
                rela_weight.sum(1, keepdim=True)  # normalize to 1
        # batch * att_size * att_feat_size
        node_feats_ = node_feats.view(-1, node_size, node_feats.size(-1))
        rela_feats_ = rela_feats.view(-1, rela_size, rela_feats.size(-1))
        node_res_ = torch.bmm(node_weight.unsqueeze(1), node_feats_).squeeze(1)
        rela_res_ = torch.bmm(rela_weight.unsqueeze(1), rela_feats_).squeeze(1)

        node_res = self.node_gate(torch.cat([node_res, rela_res], -1))
        rela_res = self.rela_gate(torch.cat([rela_res, node_res], -1))
        return node_res, rela_res


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        # maximum sample length
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.obj_voc_size = opt.obj_voc_size
        self.rela_voc_size = opt.rela_voc_size
        self.geometry_size = opt.geometry_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability
        # +1 for index 0,
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.att_feat_size, self.rnn_size),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))
        self.obj_embedding = nn.Sequential(nn.Embedding(self.obj_voc_size+1, self.input_encoding_size, padding_idx=0),
                                           nn.ReLU(),
                                           nn.Dropout(self.drop_prob_lm))
        self.rela_embedding = nn.Sequential(nn.Embedding(self.rela_voc_size+1, self.input_encoding_size, padding_idx=0),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))

        self.geometry_embedding = nn.Sequential(nn.Linear(self.geometry_size, self.rnn_size),
                                                nn.ReLU(),
                                                nn.Dropout(self.drop_prob_lm))

        self.s_gcnn = GRCNN(self.rnn_size, self.rnn_size, 3)

        self.v_gcnn = GRCNN(self.rnn_size, self.rnn_size, 3)

        # 最终输出层，只用1层全连接
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(
            ), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        # 计算attention用的
        # self.node2merge = nn.Sequential(nn.Linear(self.rnn_size*2, self.rnn_size),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(self.drop_prob_lm))

        self.node2att = nn.Linear(2*self.rnn_size, self.att_hid_size)

        # self.rela2merge = nn.Sequential(nn.Linear(self.rnn_size*2, self.rnn_size),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(self.drop_prob_lm))

        self.rela2att = nn.Linear(2*self.rnn_size, self.att_hid_size)

        self.init_weight()
        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [
            int(k) for k, v in self.vocab.items() if v in bad_endings]

    def init_weight(self):
        initrange = 0.1
        self.embed[0].weight.data.uniform_(-initrange, initrange)
        self.obj_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.rela_embedding[0].weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        obj_label, att_masks = self.clip_att(obj_label, att_masks)

        rela_label, rela_masks = self.clip_att(rela_label, rela_masks)
        geometry, rela_masks = self.clip_att(geometry, rela_masks)
        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        # wrapper后，补0的位置通过映射后还是0
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        obj2vec = pack_wrapper(self.obj_embedding, obj_label, att_masks)
        # obj2vec = self.obj_embedding(obj_label)

        rela2vec = pack_wrapper(self.rela_embedding, rela_label, rela_masks)
        # rela2vec = self.rela_embedding(rela_label)

        geometry_feats = pack_wrapper(
            self.geometry_embedding, geometry, rela_masks)
        # geometry_feats = self.geometry_embedding(geometry)
        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, obj2vec, rela2vec, geometry_feats, att_masks, rela_masks

    def _forward(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                 adj1, adj2, adj3, rela_masks, seq, att_masks):

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(
            batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, p_obj2vec, p_rela2vec, p_geometry, p_att_masks, p_rela_masks = self._prepare_feature(
            fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        if self.use_gcn:
            gcn_obj2vec, gcn_rela2vec = self.s_gcnn(
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        node_feats = torch.cat((gcn_att_feats, gcn_obj2vec), 2)
        rela_feats = torch.cat((gcn_geometry, gcn_rela2vec), 2)

        # node_feats = pack_wrapper(self.node2merge, node_feats2, p_att_masks)
        # rela_feats = pack_wrapper(self.rela2merge, rela_feats2, p_rela_masks)

        p_node_feats = pack_wrapper(self.node2att, node_feats, p_att_masks)
        p_rela_feats = pack_wrapper(self.rela2att, rela_feats, p_rela_masks)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach())
                    it.index_copy_(0, sample_ind, torch.multinomial(
                        prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(
                it, p_fc_feats, node_feats, p_node_feats, rela_feats, p_rela_feats,  p_att_masks, p_rela_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, node_feats, p_node_feats, rela_feats, p_rela_feats, att_masks, rela_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(
            xt, fc_feats, node_feats, p_node_feats, rela_feats, p_rela_feats, state, att_masks, rela_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                     adj1, adj2, adj3, rela_masks, seq, att_masks, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, p_obj2vec, p_rela2vec, p_geometry, p_att_masks, p_rela_masks = self._prepare_feature(
            fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks)

        if self.use_gcn:
            gcn_obj2vec, gcn_rela2vec = self.s_gcnn(
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        node_feats = torch.cat((gcn_att_feats, gcn_obj2vec), 2)
        rela_feats = torch.cat((gcn_geometry, gcn_rela2vec), 2)

        # node_feats = pack_wrapper(self.node2merge, node_feats2, p_att_masks)
        # rela_feats = pack_wrapper(self.rela2merge, rela_feats2, p_rela_masks)

        p_node_feats = pack_wrapper(self.node2att, node_feats, p_att_masks)
        p_rela_feats = pack_wrapper(self.rela2att, rela_feats, p_rela_masks)

        assert beam_size <= self.vocab_size + \
            1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k +
                                      1].expand(beam_size, p_fc_feats.size(1))
            # tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            # tmp_obj2vec = p_obj2vec[k:k+1].expand(*((beam_size,)+p_obj2vec.size()[1:])).contiguous()
            # tmp_rela2vec = p_rela2vec[k:k+1].expand(*((beam_size,)+p_rela2vec.size()[1:])).contiguous()
            # tmp_geometry = p_geometry[k:k+1].expand(*((beam_size,)+p_geometry.size()[1:])).contiguous()
            tmp_node_feats = node_feats[k:k+1].expand(
                *((beam_size,)+node_feats.size()[1:])).contiguous()
            tmp_p_node_feats = p_node_feats[k:k+1].expand(
                *((beam_size,)+p_node_feats.size()[1:])).contiguous()
            tmp_rela_feats = rela_feats[k:k+1].expand(
                *((beam_size,)+rela_feats.size()[1:])).contiguous()
            tmp_p_rela_feats = p_rela_feats[k:k+1].expand(
                *((beam_size,)+p_rela_feats.size()[1:])).contiguous()

            tmp_rela_masks = p_rela_masks[k:k+1].expand(
                *((beam_size,)+p_rela_masks.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(
                *((beam_size,)+p_att_masks.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                    it, tmp_fc_feats, tmp_node_feats, tmp_p_node_feats, tmp_rela_feats, tmp_p_rela_feats, tmp_att_masks, tmp_rela_masks, state)

            self.done_beams[k] = self.beam_search(
                state, logprobs, tmp_fc_feats, tmp_node_feats, tmp_p_node_feats, tmp_rela_feats, tmp_p_rela_feats, tmp_att_masks, tmp_rela_masks, opt=opt)
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                adj1, adj2, adj3, rela_masks, seq, att_masks, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                                     adj1, adj2, adj3, rela_masks, seq, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, p_obj2vec, p_rela2vec, p_geometry, p_att_masks, p_rela_masks = self._prepare_feature(
            fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks)

        if self.use_gcn:
            gcn_obj2vec, gcn_rela2vec = self.s_gcnn(
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        node_feats = torch.cat((gcn_att_feats, gcn_obj2vec), 2)
        rela_feats = torch.cat((gcn_geometry, gcn_rela2vec), 2)

        # node_feats = pack_wrapper(self.node2merge, node_feats2, p_att_masks)
        # rela_feats = pack_wrapper(self.rela2merge, rela_feats2, p_rela_masks)

        p_node_feats = pack_wrapper(self.node2att, node_feats, p_att_masks)
        p_rela_feats = pack_wrapper(self.rela2att, rela_feats, p_rela_masks)

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros(
            (batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                it, p_fc_feats, node_feats, p_node_feats, rela_feats, p_rela_feats,  p_att_masks, p_rela_masks, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(
                    seq[:, t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Impossible to generate remove_bad_endings
                tmp[torch.from_numpy(prev_bad.astype(
                    'uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t-3:t-1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(
                    ), prev_two_batch[i][1].item())
                    current = seq[i][t-1]
                    if t == 3:  # initialize
                        # {LongTensor: list containing 1 int}
                        trigrams.append({prev_two: [current]})
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t-2:t]
                # batch_size x vocab_size
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(
                    ), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                # ln(1/2) * alpha (alpha -> infty works best)
                logprobs = logprobs + (mask * -0.693 * alpha)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(
                logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs


class GRCNN(nn.Module):
    def __init__(self, node_dim=512, rela_dim=512, step=3):
        super(GRCNN, self).__init__()
        self.node_dim = node_dim
        self.rela_dim = rela_dim
        self.feat_update_step = 3
        # step1-3的参数
        self.node2node_transform = nn.ModuleList()
        self.node2rela_transform = nn.ModuleList()
        self.rela_transform = nn.ModuleList()

        for i in range(self.feat_update_step):
            self.node2node_transform.append(
                nn.ModuleList())
            self.node2rela_transform.append(
                nn.ModuleList())
            self.rela_transform.append(
                nn.ModuleList())

        # step1-3分别要3阶，2阶，1阶邻居
        self.node2node_transform[0].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[0].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[0].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[1].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[1].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[2].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        for i in range(self.feat_update_step):
            # node-rela
            self.node2rela_transform[i].append(
                nn.Linear(self.rela_dim, self.node_dim)
            )
            # rela-sub
            self.rela_transform[i].append(
                nn.Linear(self.rela_dim, self.rela_dim)
            )
            # rela-obj
            self.rela_transform[i].append(
                nn.Linear(self.rela_dim, self.rela_dim)
            )
        # self._init_weight()

    def forward(self, node, rela, p_att_masks, p_rela_masks, *adj):
        adj1, adj2, adj3, rela_sub, rela_obj, rela_n2r = adj[
            0], adj[1], adj[2], adj[3], adj[4], adj[5]

        # step1
        # num1 = torch.sum(adj1, 2, keepdim=True).cuda()  # 每个节点的一阶邻居个数
        # mask1 = torch.gt(num1, 0).float().cuda()
        # neighbors1_feat = torch.tensor(
        #     [1.]).cuda()/(num1+1e-8)*mask1*self.node_transform[0](torch.bmm(adj1, node))
        # I11 = torch.eye(adj1.shape[1]).unsqueeze(0).expand_as(adj1)
        adj11_hat = adj1  # +I11.cuda()
        D11 = torch.diag_embed(torch.sum(adj11_hat, dim=2))
        # D = torch.diag()
        neighbors11_feat = pack_wrapper(
            self.node2node_transform[0][0], torch.bmm(torch.bmm(self.inv(D11), adj11_hat), node), p_att_masks)
        # neighbors11_feat = pack_wrapper(
        #     self.node2node_transform[0][0], torch.bmm(adj11_hat, node), p_att_masks)

        # I12 = torch.eye(adj2.shape[1]).unsqueeze(0).expand_as(adj2)
        adj12_hat = adj2  # +I12.cuda()
        D12 = torch.diag_embed(torch.sum(adj12_hat, dim=2))
        neighbors12_feat = pack_wrapper(
            self.node2node_transform[0][1], torch.bmm(
                torch.bmm(self.inv(D12), adj12_hat), node), p_att_masks)
        # neighbors12_feat = pack_wrapper(
        #     self.node2node_transform[0][1],
        #     torch.bmm(adj12_hat, node), p_att_masks)

        # I13 = torch.eye(adj3.shape[1]).unsqueeze(0).expand_as(adj3)
        adj13_hat = adj3  # +I13.cuda()
        D13 = torch.diag_embed(torch.sum(adj13_hat, dim=2))
        neighbors13_feat = pack_wrapper(
            self.node2node_transform[0][2], torch.bmm(
                torch.bmm(self.inv(D13), adj13_hat), node), p_att_masks)
        # neighbors13_feat = pack_wrapper(
        #     self.node2node_transform[0][2],
        #     torch.bmm(adj13_hat, node), p_att_masks)

        node2rela_feat1 = pack_wrapper(
            self.node2rela_transform[0][0], torch.bmm(rela_n2r, rela), p_att_masks)

        node_step1 = F.relu(node + neighbors11_feat +
                            neighbors12_feat+neighbors13_feat + node2rela_feat1)

        rela_sub_feat1 = pack_wrapper(
            self.rela_transform[0][0], torch.bmm(rela_sub, node), p_rela_masks)
        rela_obj_feat1 = pack_wrapper(
            self.rela_transform[0][1], torch.bmm(rela_obj, node), p_rela_masks)

        rela_step1 = F.relu(rela+rela_sub_feat1+rela_obj_feat1)
        # step 1 end

        # step 2
        # I21 = torch.eye(adj1.shape[1]).unsqueeze(0).expand_as(adj1)
        adj21_hat = adj1  # +I21.cuda()
        D21 = torch.diag_embed(torch.sum(adj21_hat, dim=2))
        # D = torch.diag()
        neighbors21_feat = pack_wrapper(
            self.node2node_transform[1][0], torch.bmm(torch.bmm(self.inv(D21), adj21_hat), node_step1), p_att_masks)

        # I22 = torch.eye(adj2.shape[1]).unsqueeze(0).expand_as(adj2)
        adj22_hat = adj2  # +I22.cuda()
        D22 = torch.diag_embed(torch.sum(adj22_hat, dim=2))
        neighbors22_feat = pack_wrapper(
            self.node2node_transform[1][1], torch.bmm(
                torch.bmm(self.inv(D22), adj22_hat), node_step1), p_att_masks)

        node2rela_feat2 = pack_wrapper(
            self.node2rela_transform[1][0], torch.bmm(rela_n2r, rela_step1), p_att_masks)

        node_step2 = F.relu(node_step1 + neighbors21_feat +
                            neighbors22_feat + node2rela_feat2)

        rela_sub_feat2 = pack_wrapper(
            self.rela_transform[1][0], torch.bmm(rela_sub, node_step1), p_rela_masks)
        rela_obj_feat2 = pack_wrapper(
            self.rela_transform[1][1], torch.bmm(rela_obj, node_step1), p_rela_masks)

        rela_step2 = F.relu(rela_step1+rela_sub_feat2+rela_obj_feat2)
        # step 2 end

        # step 3
        # I31 = torch.eye(adj1.shape[1]).unsqueeze(0).expand_as(adj1)
        adj31_hat = adj1  # +I31.cuda()
        D31 = torch.diag_embed(torch.sum(adj31_hat, dim=2))
        # D = torch.diag()
        neighbors31_feat = pack_wrapper(
            self.node2node_transform[2][0], torch.bmm(torch.bmm(self.inv(D31), adj31_hat), node_step2), p_att_masks)

        node2rela_feat3 = pack_wrapper(
            self.node2rela_transform[2][0], torch.bmm(rela_n2r, rela_step2), p_att_masks)

        node_step3 = F.relu(node_step2 + neighbors31_feat + node2rela_feat3)

        rela_sub_feat3 = pack_wrapper(
            self.rela_transform[2][0], torch.bmm(rela_sub, node_step2), p_rela_masks)
        rela_obj_feat3 = pack_wrapper(
            self.rela_transform[2][1], torch.bmm(rela_obj, node_step2), p_rela_masks)

        rela_step3 = F.relu(rela_step2+rela_sub_feat3+rela_obj_feat3)

        return node_step3, rela_step3

    def _init_weight(self):
        for n, w in self.named_parameters():
            if n.find('weight') != -1:
                w.data.normal_(0.0, 0.01)
            elif n.find('bias') != -1:
                w.data.fill_(0.0)

    def inv(self, A, eps=1e-10):

        assert len(A.shape) == 3 and \
            A.shape[1] == A.shape[2]
        n = A.shape[1]
        U = A.clone().data
        L = A.new_zeros(A.shape).data
        L[:, range(n), range(n)] = 1
        I = L.clone()

        # A = LU
        # [A I] = [LU I] -> [U L^{-1}]
        L_inv = I
        for i in range(n-1):
            L[:, i+1:, i:i+1] = U[:, i+1:, i:i+1] / (U[:, i:i+1, i:i+1] + eps)
            L_inv[:, i+1:, :] = L_inv[:, i+1:, :] - \
                L[:, i+1:, i:i+1].matmul(L_inv[:, i:i+1, :])
            U[:, i+1:, :] = U[:, i+1:, :] - \
                L[:, i+1:, i:i+1].matmul(U[:, i:i+1, :])

        # [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
        A_inv = L_inv
        for i in range(n-1, -1, -1):
            A_inv[:, i:i+1, :] = A_inv[:, i:i+1, :] / \
                (U[:, i:i+1, i:i+1] + eps)
            U[:, i:i+1, :] = U[:, i:i+1, :] / (U[:, i:i+1, i:i+1] + eps)

            if i > 0:
                A_inv[:, :i, :] = A_inv[:, :i, :] - \
                    U[:, :i, i:i+1].matmul(A_inv[:, i:i+1, :])
                U[:, :i, :] = U[:, :i, :] - \
                    U[:, :i, i:i+1].matmul(U[:, i:i+1, :])

        A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
        return A_inv + A_inv_grad - A_inv_grad.data


class MNGrcnnCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(MNGrcnnCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(
            opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(
            opt.rnn_size * 5, opt.rnn_size)  # h^1_t, \hat v 1024, \hat rela 1024
        self.attention = Attention(opt)
        # self.init_weight()

    def forward(self, xt, fc_feats, node_feats, p_node_feats, rela_feats, p_rela_feats, state, att_masks=None, rela_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(
            att_lstm_input, (state[0][0], state[1][0]))

        # h_att； 融合过的节点特征，融合过的边特征
        att, rela = self.attention(
            h_att, node_feats, p_node_feats, rela_feats, p_rela_feats, att_masks, rela_masks)

        lang_lstm_input = torch.cat([att, rela, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(
            lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

    def init_weight(self):
        nn.init.orthogonal_(self.att_lstm.weight_hh,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.att_lstm.weight_ih,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.att_lstm.bias_hh, 0)
        nn.init.constant_(self.att_lstm.bias_ih, 0)

        nn.init.orthogonal_(self.lang_lstm.weight_hh,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.lang_lstm.weight_ih,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.lang_lstm.bias_hh, 0)
        nn.init.constant_(self.lang_lstm.bias_ih, 0)


class MNGrcnn(AttModel):
    def __init__(self, opt):
        super(MNGrcnn, self).__init__(opt)
        self.num_layers = 2
        self.use_gcn = opt.use_gcn
        self.core = MNGrcnnCore(opt)
