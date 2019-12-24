import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                adj1, adj2, adj3, labels, masks, att_masks, rela_masks,
                gts, gt_indices, sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                                        adj1, adj2, adj3, rela_masks, labels, att_masks), labels[:, 1:], masks[:, 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                                           adj1, adj2, adj3, rela_masks, labels, att_masks,
                                           mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, obj_label, rela_label, rela_sub, rela_obj, rela_n2r, geometry,
                                                     adj1, adj2, adj3, rela_masks, labels, att_masks,
                                                     opt={'sample_method': 'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(
                greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out
