import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from .vit import *
import pdb

class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='/raid2/yue/ReID/vision_language/VTB/VTB2/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(768, dim)

        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
       
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

        self.get_fc=nn.Linear(129, 1)
        

    def forward(self, imgs, word_vec, label=None):
        features = self.vit(imgs)
        
        #pdb.set_trace()
        # features1 = features.permute(0, 2, 1)  # NLD -> LND
        # logits0 = torch.cat([self.get_fc(features1[:, i, :]) for i in range(768)], dim=1)
        # logits0=logits0@(word_vec.t())

        word_embed = self.word_embed(word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        
        tex_embed = word_embed + self.tex_embed       
        vis_embed = features + self.vis_embed

        x = torch.cat([tex_embed, vis_embed], dim=1)
        #pdb.set_trace()
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        

        return logits,x