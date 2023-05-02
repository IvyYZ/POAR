import ast
from typing_extensions import Self
import numpy as np
import torch
import clip,clipS
from tqdm import tqdm
from pkg_resources import packaging
from test.classifierWeights import zeroshot_classifier,text_classfier_weights,get_pedestrian_metrics,get_pedestrian_metrics0
from data.pre_cls_pa100k import pa100kbaseDataset
from data.pre_peta_random import petabaseDataset
import pdb
import copy
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from data.text_image_dm import TextImageDataModule,TextImageDataset
from models import CustomCLIPWrapper
from torch.utils.data import Dataset,DataLoader
import ast

import os
import pdb
import pickle
import torch.nn.functional as F


os.environ['TORCH-HOME']='/raid2/yue/torch-model'
device = "cuda" if torch.cuda.is_available() else "cpu"
k_value={"hair":0,"age":1,"gender":2,"carry":3,"accessory":4, "foot":5, "upperbody":6, "lowerbody":7} #peta
#k_value={"gender":0,"age":1,"body":2,"accessory":3,"carry":4,"upperbody":5, "lowerbody":6, "foot":7}  #pa100k
#k_value={"gender":2,"age":1,"body":2,"accessory":4,"carry":3,"upperbody":6, "lowerbody":7, "foot":5}  #peta->pa100k
thres={}


def load_model(hparams,model_path):
     #加载模型
    print("Torch version:", torch.__version__)
    clip.available_models()    
    clp, _ = clip.load("ViT-B/16", device=device)
 
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
    model.eval()
    
    pdb.set_trace()
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint.state_dict()) #23.3.18
    model2=model.model.cuda()
    model2.eval()
    
    return model2

def load_peta(model):
    keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    petadata=petabaseDataset(root_path)
    classes=petadata.classes
    templates=petadata.templates
     
    zeroshot_weights=text_classfier_weights(keys,classes,templates,model)
    
    return zeroshot_weights, keys, petadata

def load_pa100k(model):
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    keys=["gender","age","body","accessory","carry","upperbody", "lowerbody", "foot"]
    data=pa100kbaseDataset(root_path)
    classes=data.classes
    templates=data.templates
    zeroshot_weights=text_classfier_weights(keys,classes,templates,model)
    return zeroshot_weights, keys,data

def load_data(hparams,model):
    root_dir="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset"
    if hparams.testset=="PA100K":
        test_root=os.path.join(root_dir,"PA-100K/PA100k_test/")
        zeroshot_weights, keys,data=load_pa100k(model)
        
    elif hparams.testset=="PA100KTrain":
        test_root=os.path.join(root_dir,"PA-100K/PA100k_train_label/")
        zeroshot_weights, keys,data=load_pa100k(model)
       
    elif hparams.testset=="PETATrain":
        test_root=os.path.join(root_dir,"PETAdata/PETA_train_label/")
        zeroshot_weights, keys,data=load_peta(model)
        
    else: #petatest
        test_root=os.path.join(root_dir,"PETAdata/PETA_select_test/")
        zeroshot_weights, keys,data=load_peta(model)
   
    test_dataset=TextImageDataset(folder=test_root, image_size=hparams.imgSize, batch_size=hparams.minibatch_size)
    dataloader=DataLoader(dataset=test_dataset, batch_size=hparams.minibatch_size, shuffle=False)
    return dataloader,zeroshot_weights, keys,data

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys): 
    classes=data.classes
    gtlabels=data.labels
    pred_logits=np.zeros((data.test_size,data.test_label_num))   #(90000,26) train
    labels=np.zeros((data.test_size,data.test_label_num)) 
    with torch.no_grad():
        logits_all={}
        t_num=0

        for i, (image_tensor, description, name, label) in enumerate(dataloader):                               
            #imgs=preprocess(Image.open(imgpath+name).convert('RGB'))
            for la in label:
                aa=ast.literal_eval(la)
                labels[t_num]=aa[:data.test_label_num]
                t_num+=1
            image_tensor=image_tensor.cuda()
            image_features, attention = model.encode_image(image_tensor,k_num=8)
            for item in keys:
                ite=item.split("_")[0]
                k=int(k_value[ite])   
                #pdb.set_trace()
                image_features[:,k,:] /= image_features[:,k,:].norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * image_features[:,k,:] @ zeroshot_weights[item]

                logits=logits.cpu()
                if item in logits_all.keys():
                    logits_all[item]=torch.cat((logits_all[item],logits),0)
                else:
                    logits_all[item]=logits
        
        with open('train_logits.pkl', 'wb') as f:
            pickle.dump({"logits_all": logits_all,"labels":labels}, f)
        
        pred_logits=convert_logits(hparams,keys,classes,gtlabels,logits_all,pred_logits)     

    pred_label=copy.deepcopy(pred_logits)
    Traverse_threshold(pred_logits,labels,pred_label)
                
def convert_logits(hparams,keys,classes,gtlabels,logits_all,pred_logits):
    if "PA100K" in hparams.testset:
        for item in keys:
                print("item",item)
                kn=item.split("_")            
                itemC=item
                ite=item.split("_")[0]
                
                for jth in range(len(classes[itemC])):               
                    catg=classes[itemC][jth]
                    if catg in gtlabels[ite].keys():
                        index_t=gtlabels[ite][catg]
                        pred_logits[:,index_t]=(logits_all[item].numpy())[:,jth]  
    else:
         for item in keys:
            print("item",item)
            kn=item.split("_")
            ite=item.split("_")[0]
            if len(kn)>1:
                if kn[1]=='2':
                    itemC="color"                       
                elif kn[1]=='3':
                    itemC="style"                  
                else:
                    itemC=kn[0]             
            else:
                itemC=item

            for jth in range(len(classes[itemC])):               
                catg=classes[itemC][jth]
                if catg in gtlabels[ite].keys():
                    index_t=gtlabels[ite][catg]
                    pred_logits[:,index_t]=(logits_all[item].numpy())[:,jth]    

    return pred_logits


def get_class(item,classes):
    kn=item.split("_")
    k1=kn[0]
    if len(kn) > 1:
        k2 = item.split("_")[1]
        if k2=='1':
            cls=classes[k1]
        elif k2=='2':                    
            cls=classes["color"]
        else:
            cls=classes["style"]
    else:
        cls=classes[k1]
    return cls

def Traverse_threshold(logits,labels,pred_labels):
    thres={}
    accuracy={}
    labels=np.array(labels)
    # with open('results_peta_train.pkl', 'wb') as f:  
    #     pickle.dump({"logits": logits,"labels":labels}, f)
    #pdb.set_trace()
    for i in range(len(logits[0])):
        print(i,"-th col start search----")
        sort_list=np.sort(logits[:,i])
        d1=labels[:,i:i+1]  
        d2=logits[:,i:i+1]
        a1=0
        t1=0
        for thre in sort_list:              
            a3,_=get_pedestrian_metrics(d1, d2,threshold=thre)
            if a3.label_acc>a1:            #label_acc,add_acc 
                a1=a3.label_acc 
                a2=copy.deepcopy(a3)
                t1=thre
        print("thres[",i,"] is:", t1)
        print("the best label_acc is:",a2.label_acc)
        thres[i]=t1
        accuracy[i]=a2
        a3,pred_label_best=get_pedestrian_metrics(d1, d2,threshold=t1)
        pred_labels[:,i:i+1]=pred_label_best
 
    acc=get_pedestrian_metrics0(labels, pred_labels)  
    print(acc)     
    pdb.set_trace()
    return acc

def main(hparams):

    #model_path="./lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt"     \
    #model_path="./lightning_logs/version_3/checkpoints/epoch=14-step=5084.ckpt" 
    model_path="/home/xiaodui/zy/PAR/TS/lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt" 
    #model_path="/home/xiaodui/zy/PAR/TS/stmodels/model_train_peta/model1.pth"

    #加载模型
    #pdb.set_trace()
    model=load_model(hparams,model_path)
    #加载数据
    dataloader,zeroshot_weights, keys,data=load_data(hparams,model)
    #测试数据  
    get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--testset', type=str, required=True, help='[PA100K,PA100KTrain,PETA,PETATrain]')
    parser.add_argument('--imgSize', type=int, default=224, help='input image size')
    args = parser.parse_args()

    main(args)