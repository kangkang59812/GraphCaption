from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
import torchvision
from .CaptionModel import CaptionModel
from collections import OrderedDict
bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with',
               'before', 'after', 'on', 'upon', 'near', 'to', 'is', 'are', 'am']
bad_endings += ['the']


class AttModel(CaptionModel):
    def __init__(self, opt, encoded_image_size=14, K=20, L=1024):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        # maximum sample length
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length

        self.att_hid_size = opt.att_hid_size

        self.ss_prob = 0.0  # Schedule sampling probability
        # +1 ??
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        # 最终输出层，只用1层全连接
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(
            ), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [
            int(k) for k, v in self.vocab.items() if v in bad_endings]

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def _forward(self, attrs, imgs, seq):
        batch_size = imgs.size(0)
        state = self.init_hidden(batch_size)

        outputs = imgs.new_zeros(
            batch_size, seq.size(1) - 1, self.vocab_size+1)
        attributes, imgs_features = attrs, imgs

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample

                sample_prob = imgs.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
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
                it, attributes, imgs_features, state, i)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, attributes, imgs_features, state, ii=0):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(
            xt, attributes, imgs_features, state, ii)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, attrs, imgs, seq, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = imgs.size(0)

        assert beam_size <= self.vocab_size + \
            1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity
        attributes, imgs_features = attrs, imgs

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            tmp_attributes = attributes[k:k +
                                        1].expand(beam_size, attributes.size(1))
            tmp_imgs_features = imgs_features[k:k+1].expand(
                *((beam_size,)+imgs_features.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = imgs.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                    it, tmp_attributes, tmp_imgs_features, state, t)

            self.done_beams[k] = self.beam_search(
                state, logprobs, tmp_attributes, tmp_imgs_features, opt=opt)
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, attrs, imgs, seq, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(attrs, imgs, opt)

        batch_size = imgs.size(0)
        state = self.init_hidden(batch_size)

        attributes, imgs_features = attrs, imgs

        trigrams = []  # will be a list of batch_size dictionaries

        seq = imgs.new_zeros(
            (batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = imgs.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = imgs.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                it, attributes, imgs_features, state, t)

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
                #logprobs = logprobs - (mask * 1e9)
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


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.v2att = nn.Linear(2048, self.att_hid_size)
        #self.a2att = nn.Linear(1024, self.att_hid_size)
        self.hatt2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.hvis2att = nn.Linear(self.rnn_size, self.att_hid_size)
        # self.alpha_net1 = nn.Linear(self.att_hid_size, 1)

        self.full_att = nn.Linear(self.att_hid_size, 1)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, h_att, prev_h2, imgs_features):

        # The p_att_feats here is already projected
        att_size = imgs_features.numel() // imgs_features.size(0) // imgs_features.size(-1)
        p_imgs_features = self.v2att(
            imgs_features).view(-1, att_size, self.att_hid_size)

        # p_attributes = self.a2att(attributes).unsqueeze(
        #     1).expand_as(p_imgs_features)
        
        # batch * att_hid_size
        att_hatt = self.hatt2att(h_att)
        # batch * att_size * att_hid_size
        att_hatt = att_hatt.unsqueeze(1).expand_as(p_imgs_features)
        # batch * att_size * att_hid_size

        # batch * att_hid_size
        att_hvis = self.hvis2att(prev_h2)
        # batch * att_size * att_hid_size
        att_hvis = att_hvis.unsqueeze(1).expand_as(p_imgs_features)
        # batch * att_size * att_hid_size

        dot = p_imgs_features + att_hatt + att_hvis
        # batch * att_size * att_hid_size
        # dot1 = torch.tanh(dot)
        # # (batch * att_size) * att_hid_size
        # dot1 = dot1.view(-1, self.att_hid_size)
        # # (batch * att_size) * 1
        # dot1 = self.alpha_net1(dot1)
        # dot1 = dot1.view(-1, att_size)

        # # batch * att_size
        # weight1 = F.softmax(dot1, dim=1)

        # # batch * att_size * att_feat_size
        # imgs_features_ = imgs_features.view(-1,
        #                                     att_size, imgs_features.size(-1))
        # imgs_features_res = torch.bmm(
        #     weight1.unsqueeze(1), imgs_features_).squeeze(1)

        # result = torch.cat([imgs_features_res, attributes], dim=1)
        att = self.full_att(self.dropout(
            self.relu(dot))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels, 1)
        attention_weighted_encoding = (
            imgs_features * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding


class PLSTMCore(nn.Module):
    def __init__(self, opt):
        super(PLSTMCore, self).__init__()
        self.opt = opt
        self.drop_prob_lm = opt.drop_prob_lm
        self.init_x0 = nn.Linear(1024, opt.input_encoding_size)

        self.attrs_lstm = nn.LSTMCell(
            opt.input_encoding_size, opt.rnn_size)

        self.visual_lstm = nn.LSTMCell(
            opt.input_encoding_size+2048, opt.rnn_size)
        self.attention = Attention(opt)
        self.f_beta = nn.Linear(opt.rnn_size, 2048)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xt, attributes, imgs_features, state, ii=0):
        prev_h1, prev_h2 = state[0][0], state[0][1]
        if ii == 0:
            tmp = self.init_x0(attributes)
            h_att0, c_att0 = self.attrs_lstm(
                tmp, (state[0][0], state[1][0]))
            attrs_lstm_input = xt
            h_att, c_att = self.attrs_lstm(
                attrs_lstm_input, (h_att0, c_att0))

            atten_input = self.attention(
                h_att, prev_h2, imgs_features)

            gate = self.sigmoid(self.f_beta(prev_h2))

            vis_lstm_input = torch.cat([gate*atten_input, xt], 1)

            h_lang, c_lang = self.visual_lstm(
                vis_lstm_input, (state[0][1], state[1][1]))
        else:
            attrs_lstm_input = xt

            h_att, c_att = self.attrs_lstm(
                attrs_lstm_input, (state[0][0], state[1][0]))
            atten_input = self.attention(
                h_att, prev_h2, imgs_features)

            gate = self.sigmoid(self.f_beta(prev_h2))

            vis_lstm_input = torch.cat([gate*atten_input, xt], 1)

            h_lang, c_lang = self.visual_lstm(
                vis_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att,  h_lang]),
                 torch.stack([c_att,  c_lang]))

        return output, state


class PLSTM(AttModel):
    def __init__(self, opt):
        super(PLSTM, self).__init__(opt)
        self.num_layers = 2
        self.core = PLSTMCore(opt)
