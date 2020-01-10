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
from .AttModel import Attention as Att
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


class GateAttention(nn.Module):
    def __init__(self, opt):
        super(GateAttention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size

        self.snodeAttention = Att(opt)
        self.srelaAttention = Att(opt)
        self.vnodeAttention = Att(opt)
        self.vrelaAttention = Att(opt)

        self.snode_gate = nn.Sequential(nn.Linear(
            self.input_encoding_size, self.input_encoding_size), nn.Sigmoid())
        self.srela_gate = nn.Sequential(nn.Linear(
            self.input_encoding_size, self.input_encoding_size), nn.Sigmoid())
        self.vnode_gate = nn.Sequential(nn.Linear(
            self.input_encoding_size, self.input_encoding_size), nn.Sigmoid())
        self.vrela_gate = nn.Sequential(nn.Linear(
            self.input_encoding_size, self.input_encoding_size), nn.Sigmoid())
        # self._init_weight()

    def forward(self, h, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, att_masks=None, rela_masks=None):

        snode_ = self.snodeAttention(h, gcn_obj2vec, snode_feats, att_masks)
        srela_ = self.srelaAttention(h, gcn_rela2vec, srela_feats, rela_masks)
        vnode_ = self.vnodeAttention(h, gcn_att_feats, vnode_feats, att_masks)
        vrela_ = self.vrelaAttention(h, gcn_geometry, vrela_feats, rela_masks)

        snode = snode_*self.snode_gate(vnode_)
        srela = srela_*self.srela_gate(vrela_)
        vnode = vnode_*self.vnode_gate(snode_)
        vrela = vrela_*self.vrela_gate(srela_)

        return snode, srela, vnode, vrela

    def _init_weight(self):
        for n, w in self.named_parameters():
            if n.find('weight') != -1:
                w.data.normal_(0.0, 0.01)
            elif n.find('bias') != -1:
                w.data.fill_(0.0)


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
        if opt.use_box:
            self.att_feat_size = self.att_feat_size + 5
        # +1 for index 0,
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.input_encoding_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.att_feat_size, self.input_encoding_size),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))
        self.obj_embedding = nn.Sequential(nn.Embedding(self.obj_voc_size+1, self.input_encoding_size, padding_idx=0),
                                           nn.ReLU(),
                                           nn.Dropout(self.drop_prob_lm))
        self.rela_embedding = nn.Sequential(nn.Embedding(self.rela_voc_size+1, self.input_encoding_size, padding_idx=0),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))

        self.geometry_embedding = nn.Sequential(nn.Linear(self.geometry_size, self.input_encoding_size),
                                                nn.ReLU(),
                                                nn.Dropout(self.drop_prob_lm))

        self.s_gcnn = GRCNN(self.input_encoding_size, self.input_encoding_size, 2)

        self.v_gcnn = GRCNN(self.input_encoding_size, self.input_encoding_size, 2)

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

        self.snode2att = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.vnode2att = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.srela2att = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.vrela2att = nn.Linear(self.input_encoding_size, self.att_hid_size)

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
        # wrapper后，通过embed该为0的 行 还是0
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
                 adj1, adj2, rela_masks, seq, att_masks):

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
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        snode_feats = pack_wrapper(self.snode2att, gcn_obj2vec, p_att_masks)
        vnode_feats = pack_wrapper(self.vnode2att, gcn_att_feats, p_att_masks)

        srela_feats = pack_wrapper(self.srela2att, gcn_rela2vec, p_rela_masks)
        vrela_feats = pack_wrapper(self.vrela2att, gcn_geometry, p_rela_masks)

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
                it, p_fc_feats, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, p_att_masks, p_rela_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, att_masks, rela_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(
            xt, fc_feats, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, state, att_masks, rela_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                     adj1, adj2, rela_masks, seq, att_masks, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, p_obj2vec, p_rela2vec, p_geometry, p_att_masks, p_rela_masks = self._prepare_feature(
            fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks)

        if self.use_gcn:
            gcn_obj2vec, gcn_rela2vec = self.s_gcnn(
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        snode_feats = pack_wrapper(self.snode2att, gcn_obj2vec, p_att_masks)
        vnode_feats = pack_wrapper(self.vnode2att, gcn_att_feats, p_att_masks)

        srela_feats = pack_wrapper(self.srela2att, gcn_rela2vec, p_rela_masks)
        vrela_feats = pack_wrapper(self.vrela2att, gcn_geometry, p_rela_masks)

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
          
            tmp_gcn_obj2vec = gcn_obj2vec[k:k+1].expand(
                *((beam_size,)+gcn_obj2vec.size()[1:])).contiguous()
            tmp_snode_feats = snode_feats[k:k+1].expand(
                *((beam_size,)+snode_feats.size()[1:])).contiguous()

            tmp_gcn_rela2vec = gcn_rela2vec[k:k+1].expand(
                *((beam_size,)+gcn_rela2vec.size()[1:])).contiguous()
            tmp_srela_feats = srela_feats[k:k+1].expand(
                *((beam_size,)+srela_feats.size()[1:])).contiguous()

            tmp_gcn_att_feats = gcn_att_feats[k:k+1].expand(
                *((beam_size,)+gcn_att_feats.size()[1:])).contiguous()
            tmp_vnode_feats = vnode_feats[k:k+1].expand(
                *((beam_size,)+vnode_feats.size()[1:])).contiguous()

            tmp_gcn_geometry = gcn_geometry[k:k+1].expand(
                *((beam_size,)+gcn_geometry.size()[1:])).contiguous()
            tmp_vrela_feats = vrela_feats[k:k+1].expand(
                *((beam_size,)+vrela_feats.size()[1:])).contiguous()

            tmp_rela_masks = p_rela_masks[k:k+1].expand(
                *((beam_size,)+p_rela_masks.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(
                *((beam_size,)+p_att_masks.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                    it, tmp_fc_feats, tmp_gcn_obj2vec, tmp_snode_feats, tmp_gcn_rela2vec, tmp_srela_feats,  tmp_gcn_att_feats, tmp_vnode_feats, tmp_gcn_geometry, tmp_vrela_feats, tmp_att_masks, tmp_rela_masks, state)

            self.done_beams[k] = self.beam_search(
                state, logprobs, tmp_fc_feats, tmp_gcn_obj2vec, tmp_snode_feats, tmp_gcn_rela2vec, tmp_srela_feats,  tmp_gcn_att_feats, tmp_vnode_feats, tmp_gcn_geometry, tmp_vrela_feats, tmp_att_masks, tmp_rela_masks, opt=opt)
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                adj1, adj2, rela_masks, seq, att_masks, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                                     adj1, adj2, rela_masks, seq, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, p_obj2vec, p_rela2vec, p_geometry, p_att_masks, p_rela_masks = self._prepare_feature(
            fc_feats, att_feats, obj_label, rela_label, geometry, att_masks, rela_masks)

        if self.use_gcn:
            gcn_obj2vec, gcn_rela2vec = self.s_gcnn(
                p_obj2vec, p_rela2vec, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)

            gcn_att_feats, gcn_geometry = self.v_gcnn(
                p_att_feats, p_geometry, p_att_masks, p_rela_masks, adj1, adj2, rela_sub, rela_obj, rela_n2r)
        else:
            gcn_obj2vec, gcn_rela2vec = p_obj2vec, p_rela2vec
            gcn_att_feats, gcn_geometry = p_att_feats, p_geometry

        snode_feats = pack_wrapper(self.snode2att, gcn_obj2vec, p_att_masks)
        vnode_feats = pack_wrapper(self.vnode2att, gcn_att_feats, p_att_masks)

        srela_feats = pack_wrapper(self.srela2att, gcn_rela2vec, p_rela_masks)
        vrela_feats = pack_wrapper(self.vrela2att, gcn_geometry, p_rela_masks)

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros(
            (batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                it, p_fc_feats, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, p_att_masks, p_rela_masks, state)

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
    def __init__(self, node_dim=512, rela_dim=512, step=2):
        super(GRCNN, self).__init__()
        self.node_dim = node_dim
        self.rela_dim = rela_dim
        self.feat_update_step = step
        # step1-3的参数
        self.node2node_transform = nn.ModuleList()
        self.node2rela_transform = nn.ModuleList()
        self.rela_transform = nn.ModuleList()
        self.p = 0.
        for i in range(self.feat_update_step):
            self.node2node_transform.append(
                nn.ModuleList())
            self.node2rela_transform.append(
                nn.ModuleList())
            self.rela_transform.append(
                nn.ModuleList())

        # step1-3分别要2阶，1阶邻居
        self.node2node_transform[0].append(
            nn.Linear(self.node_dim, self.node_dim)
        )
        self.node2node_transform[0].append(
            nn.Linear(self.node_dim, self.node_dim)
        )

        self.node2node_transform[1].append(
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
        adj1, adj2, rela_sub, rela_obj, rela_n2r = adj[
            0], adj[1], adj[2], adj[3], adj[4]

        # step1
        # num1 = torch.sum(adj1, 2, keepdim=True).cuda()  # 每个节点的一阶邻居个数
        # mask1 = torch.gt(num1, 0).float().cuda()
        # neighbors1_feat = torch.tensor([1.]).cuda()/(num1+1e-8)*mask1*self.node_transform[0](torch.bmm(adj1, node))
        # I11 = torch.eye(adj1.shape[1]).unsqueeze(0).expand_as(adj1)
        #adj11_hat = adj1  # +I11.cuda()
        #D11 = torch.diag_embed(torch.sum(adj11_hat, dim=2))
       
        # neighbors11_feat = pack_wrapper(
        #     self.node2node_transform[0][0], torch.bmm(torch.bmm(self.inv(D11), adj11_hat), node), p_att_masks)
        neighbors11_feat = pack_wrapper(
            self.node2node_transform[0][0], torch.bmm(adj1, node), p_att_masks)

        # I12 = torch.eye(adj2.shape[1]).unsqueeze(0).expand_as(adj2)
        # adj12_hat = adj2  # +I12.cuda()
        # D12 = torch.diag_embed(torch.sum(adj12_hat, dim=2))
        # neighbors12_feat = pack_wrapper(
        #     self.node2node_transform[0][1], torch.bmm(
        #         torch.bmm(self.inv(D12), adj12_hat), node), p_att_masks)
        neighbors12_feat = pack_wrapper(
            self.node2node_transform[0][1], torch.bmm(adj2, node), p_att_masks)

        node2rela_feat1 = pack_wrapper(
            self.node2rela_transform[0][0], torch.bmm(rela_n2r, rela), p_att_masks)

        node_step1 = F.dropout(F.relu(node + neighbors11_feat +
                                      neighbors12_feat + node2rela_feat1), p=self.p)

        rela_sub_feat1 = pack_wrapper(
            self.rela_transform[0][0], torch.bmm(rela_sub, node), p_rela_masks)
        rela_obj_feat1 = pack_wrapper(
            self.rela_transform[0][1], torch.bmm(rela_obj, node), p_rela_masks)

        rela_step1 = F.dropout(
            F.relu(rela+rela_sub_feat1+rela_obj_feat1), p=self.p)
        # step 1 end

        # step 2
        # I21 = torch.eye(adj1.shape[1]).unsqueeze(0).expand_as(adj1)
        # adj21_hat = adj1  # +I21.cuda()
        # D21 = torch.diag_embed(torch.sum(adj21_hat, dim=2))
        # D = torch.diag()
        # neighbors21_feat = pack_wrapper(
        #     self.node2node_transform[1][0], torch.bmm(torch.bmm(self.inv(D21), adj21_hat), node_step1), p_att_masks)
        neighbors21_feat = pack_wrapper(
             self.node2node_transform[1][0], torch.bmm(adj1, node_step1), p_att_masks)

        node2rela_feat2 = pack_wrapper(
            self.node2rela_transform[1][0], torch.bmm(rela_n2r, rela_step1), p_att_masks)

        node_step2 = F.dropout(
            F.relu(node_step1 + neighbors21_feat + node2rela_feat2), p=self.p)

        rela_sub_feat2 = pack_wrapper(
            self.rela_transform[1][0], torch.bmm(rela_sub, node_step1), p_rela_masks)
        rela_obj_feat2 = pack_wrapper(
            self.rela_transform[1][1], torch.bmm(rela_obj, node_step1), p_rela_masks)

        rela_step2 = F.dropout(
            F.relu(rela_step1+rela_sub_feat2+rela_obj_feat2), p=self.p)
        # step 2 end

        return node_step2, rela_step2

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
            opt.input_encoding_size*2 + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(
            opt.rnn_size + opt.input_encoding_size*4, opt.rnn_size)  # h^1_t, \hat v 1024, \hat rela 1024
        self.attention = GateAttention(opt)
        # self.init_weight()

    def forward(self, xt, fc_feats, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, state, att_masks=None, rela_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(
            att_lstm_input, (state[0][0], state[1][0]))

        # h_att； 融合过的节点特征，融合过的边特征
        satt, vatt, srela, vrela = self.attention(
            h_att, gcn_obj2vec, snode_feats, gcn_rela2vec, srela_feats,  gcn_att_feats, vnode_feats, gcn_geometry, vrela_feats, att_masks, rela_masks)

        lang_lstm_input = torch.cat([satt, vatt, srela, vrela, h_att], 1)
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
