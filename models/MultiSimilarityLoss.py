# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import pdb
import torch
from torch import nn
import pdb

class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0   #cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = 40.0  #cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG

    def forward(self, feats, labels,label_num):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()
        #pdb.set_trace()
        for i in range(batch_size-label_num):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
         
            neg_pair_ = sim_mat[i][labels != labels[i]]
            
            if len(pos_pair_)>0:
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            else:
                neg_pair = neg_pair_[neg_pair_ + self.margin>0.2] 
            
            if len(neg_pair_)>0:
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            else:
                pos_pair = pos_pair_[pos_pair_ - self.margin <0.8]
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
        #pdb.set_trace()
        loss = sum(loss) / batch_size
        return loss
