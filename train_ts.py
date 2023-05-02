import torch
import torch.nn as nn
from torchvision import models
import clip
import math
import random
from load_datasets import load_data
import ast
from typing_extensions import Self
import numpy as np
import torch
import clip,clipS
from tqdm import tqdm
from pkg_resources import packaging
from test.classifierWeights import zeroshot_classifier,text_classfier_weights,get_pedestrian_metrics,get_pedestrian_metrics0,text_classfier_weights_all
from data.pre_cls_pa100k import pa100kbaseDataset
from data.pre_peta_random import petabaseDataset
from data.pre_rap import parbaseDataset
import pdb
import copy
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from data.text_image_dm import TextImageDataModule,TextImageDataset
from models import CustomCLIPWrapper
from torch.utils.data import Dataset,DataLoader
import ast
from PIL import Image

import os
import pdb
import pickle
import torch.nn.functional as F

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device="cuda")
        
    def forward(self, x):
        with torch.no_grad():
            _, logits_per_image = self.clip_model(x)
        return logits_per_image
   
def load_model(hparams):
    #加载模型
    print("Torch version:", torch.__version__)
    clip.available_models()    
    clp, _ = clip.load("ViT-B/16", device="cuda")

    for p in clp.parameters(): 
        p.data = p.data.float() 
        if p.grad:
            p.grad.data = p.grad.data.float()

    model = CustomCLIPWrapper(clp.transformer, hparams.minibatch_size, avg_word_embs=True)

    model.model.token_embedding = clp.token_embedding
    model.model.ln_final = clp.ln_final
    model.model.text_projection = clp.text_projection
    model.model.positional_embedding = clp.positional_embedding
    model.model.logit_scale = clp.logit_scale
    #model.eval()
    
    checkpoint=torch.load(hparams.modelpath)
    model.load_state_dict(checkpoint['state_dict'])
    #pdb.set_trace()
    #model2=model.model.cuda()
    #model2.eval()
    
    return model


def kl_div_loss(outputs, targets, T=5):
    log_softmax_outputs = nn.functional.log_softmax(outputs/T, dim=1)
    softmax_targets = nn.functional.softmax(targets/T, dim=1)
    #return nn.KLDivLoss(reduction='batchmean')(log_softmax_outputs, softmax_targets)
    loss=nn.functional.kl_div(log_softmax_outputs,softmax_targets,size_average=False)*(T**2)/outputs.shape[0]
    return loss 

def get_mi_loss(Stmodel,image,text):
    
    for i in range(len(image)):
        debug_image = image[i].permute(1, 2, 0)
        debug_image = debug_image - debug_image.min()
        debug_image = debug_image / debug_image.max() * 255.
        
        im = Image.fromarray(np.uint8(debug_image.cpu().numpy()), mode='RGB')
        im.save(f'debug/debug_{i:02d}.jpg')

    tokenized_text, tokenized_text_split = Stmodel.tokenize(text)   
    image = image.cuda()

    # image loss
    smodel= Stmodel.model.cuda()
    image_features, attention =smodel.encode_image(image, k_num=8)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    for i in range(len(image)):
        for k in range(8):
            debug_image = attention[i][k][8:].view(14, 14)
            debug_image = debug_image - debug_image.min()
            debug_image = debug_image / debug_image.max() * 255.
            debug_image = torch.stack([debug_image, debug_image, debug_image], dim=-1)
            debug_image = debug_image.permute(2, 0, 1).unsqueeze(0)
            debug_image = F.interpolate(debug_image, scale_factor=16)[0].permute(1, 2, 0)
            im = Image.fromarray(np.uint8(debug_image.detach().cpu().numpy()), mode='RGB')
            im.save(f'debug/debug_{i:02d}_k{k:02d}_attn.jpg')

    loss_i, loss_t,loss_g = 0, 0, 0
    acc_i, acc_t, cnt_i, cnt_t = 0, 0, 0, 0
    for k in range(8):
        indices, text1, text_split2 = Stmodel.filter_nontext(tokenized_text,tokenized_text_split, k)
        if len(text1)==0:
            continue
        text_mbs,text_nolap_mbs,indices,ground_truth_T,target_classes,weight=Stmodel.get_text_label(indices, text1, text_split2, k)
        txt = Stmodel.encode_text(text_mbs)
        txt = txt / txt.norm(dim=-1, keepdim=True)

        txt_nolap = Stmodel.encode_text(text_nolap_mbs)
        txt_nolap = txt_nolap / txt_nolap.norm(dim=-1, keepdim=True)

        target_classes=target_classes.type_as(image_features).long().cuda()
        src_logits=image_features[indices][:, k] @ txt_nolap.t()
        src_logits = src_logits * Stmodel.model.logit_scale.exp()
        
        loss_ik = Stmodel.masked_out_cross_entropy(src_logits, target_classes,weight)
        loss_tk = Stmodel.masked_out_cross_entropy(src_logits.t(), target_classes.t())
        
        loss_i += loss_ik
        loss_t += loss_tk
            
        results_t=torch.argmax(src_logits, 0)
        for i in range (len(results_t)):
            if target_classes[:, i].sum() == 0: continue
            if target_classes[int(results_t[i])][i]==1:
                acc_t+=1
            cnt_t += 1

        results_i=torch.argmax(src_logits, 1)
        for i in range (len(results_i)):
            if target_classes[i][int(results_i[i])]==1:
                acc_i+=1
            cnt_i += 1

        print(" K:",k, " loss_i:",loss_ik.detach(), " loss_t:",loss_tk.detach(), "acc_i:",acc_i/cnt_i," acc_t:",acc_t/cnt_t,)
    acc_i, acc_t = acc_i / cnt_i, acc_t / cnt_t
    Stmodel.log_dict({'part': k,'loss_i': loss_i.detach() / 8,'loss_t': loss_t.detach() / 8, 'acc_i': acc_i ,'acc_t': acc_t}, prog_bar=True)

    loss = (loss_i + loss_t) / 16 #+ loss_g/8
    
    return loss, image_features
     

def train_student_model(student_model, teacher_model, dataloader, optimizer, epoch):
    student_model.train()

    for i, (image, text, name, label) in enumerate(dataloader):         
               
        #data = data.cuda()
        optimizer.zero_grad()
        # Get teacher predictions
        teacher_outputs =teacher_model.clip_model.visual(image.cuda().half()).detach() #teacher_model(image).detach()

        # Get student predictions
        #student_outputs = student_model(image)
        loss_im, student_outputs = get_mi_loss(student_model, image, text)

        # Compute KL divergence loss
        loss_kl=0
        k=7
        #pdb.set_trace()
        student_outputs = student_outputs.permute(1, 0, 2)
        print(student_outputs.shape)
        for j in range(k):
            try:
                loss_kl += kl_div_loss(student_outputs[j], teacher_outputs,T=5)
            except:
                print(i)
                pdb.set_trace()
        alpha=1
        g_loss=loss_kl*alpha+loss_im*1.0 #(1-alpha)
        print("loss_kl:",loss_kl,"loss_im:",loss_im)

        # Backward pass0+

        g_loss.backward()
        optimizer.step()

        # Print training progress
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i* len(image), len(dataloader),
                100. * i / len(dataloader), g_loss.item()))
           # pdb.set_trace()
            # model.load_state_dict(checkpoint['state_dict'])
        torch.save(student_model, './stmodels/model'+str(epoch)+".pth")


def main(hparams):
    #Set up models and optimizer
    teacher_model = TeacherModel().cuda()
    student_model = load_model(hparams).cuda()

    #set up dataset
    train_loader,keys,data=load_data(hparams,student_model)

    #init
    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

    #Train student model for 10 epochs
    for epoch in range(1, 11):
        train_student_model(student_model, teacher_model, train_loader, optimizer, epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=32)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--testset', type=str, required=True, help='[PA100K,PA100KTrain,PETA,PETATrain]')
    parser.add_argument('--imgSize', type=int, default=224, help='input image size')
    parser.add_argument('--modelpath', type=str, required=True, help='pa100kpath,petapath,rapv1path')
    args = parser.parse_args()

    main(args)