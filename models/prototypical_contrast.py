from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()

        self.temperature = 1
        self.m = 10
        self.n = 2000

    def _contrastive(self, base_pro, pos_pro, neg_dict):

        loss = torch.zeros(1).cuda()

        for base, pos in zip(base_pro, pos_pro):
            positive_dot_contrast = torch.div(F.cosine_similarity(base, pos, 0),
                                              self.temperature)
            negative_samples = neg_dict
            if negative_samples.shape[0] > self.m:
               perm = torch.randperm(negative_samples.shape[0])
               negative_samples = negative_samples[perm[:self.m]]
            negative_dot_contrast = torch.div(F.cosine_similarity(base, torch.transpose(negative_samples, 0, 1), 0),
                                              self.temperature)
            pos_logits = torch.exp(positive_dot_contrast)
            neg_logits = torch.exp(negative_dot_contrast).sum()
            mean_log_prob_pos = - torch.log((pos_logits / (neg_logits)) + 1e-8)

            loss = loss + mean_log_prob_pos.mean()

        return loss

    def forward(self, Q_feats, S_feats, Q_predit, S_labels, negative_dict):
        S_labels = S_labels.float().clone()
        S_labels = rearrange(S_labels, "b h w -> b 1 h w")
        S_labels = F.interpolate(S_labels, (S_feats.shape[2], S_feats.shape[3]), mode='nearest')
        Q_predit = torch.cat((1.0 - Q_predit, Q_predit), dim=1)
        Q_predit_pro = Weighted_GAP(Q_feats, Q_predit.max(1)[1].unsqueeze(1)).squeeze(-1)
        S_GT_pro = Weighted_GAP(S_feats, S_labels).squeeze(-1)

        if negative_dict.shape[0] > self.n:
            indices = torch.randperm(negative_dict.size(0))[:self.n]
            negative_dict = negative_dict[indices]

        loss = self._contrastive(Q_predit_pro, S_GT_pro, negative_dict)

        return loss