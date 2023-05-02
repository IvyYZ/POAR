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

import os
import pdb
import pickle
import torch.nn.functional as F


os.environ['TORCH-HOME']='/raid2/yue/torch-model'
device = "cuda" if torch.cuda.is_available() else "cpu"
thres={}
thres={}
def get_k_value(hparams):
    #pdb.set_trace()
    if "PA100K" in hparams.testset:  
        if hparams.trainset=="PETA":
            k_value={"gender":2,"age":1,"body":2,"accessory":4,"carry":3,"upperbody":6, "lowerbody":7, "foot":5}  #peta->pa100k
        else:  
            #k_value={"gender":2,"age":1,"body":2,"accessory":4,"carry":3,"upperbody":6, "lowerbody":7, "foot":5}  #RAPv1->pa100k
            k_value={"gender":0,"age":1,"body":2,"accessory":3,"carry":4,"upperbody":5, "lowerbody":6, "foot":7}  #pa100k
    elif "RAPv1" in hparams.testset: 
        if hparams.trainset=="PA100K":
            k_value={"head":2,"age":1,"gender":0,"attach":4,"action":3,"foot":7,"upperbody":5,"lowerbody":6}  #pa100k--rapv1
        else:
            k_value={"head":0,"age":1,"gender":2,"attach":3,"action":4,"foot":5,"upperbody":6,"lowerbody":7}
    else:       
        if hparams.trainset=="PA100K":
            k_value={"hair":2,"age":1,"gender":0,"carry":4,"accessory":3, "foot":7, "upperbody":5, "lowerbody":6} #pa100k->peta
        else:
            k_value={"hair":0,"age":1,"gender":2,"carry":3,"accessory":4, "foot":5, "upperbody":6, "lowerbody":7} #peta
    return k_value




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
    
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
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
    #zeroshot_weights=text_classfier_weights_all(keys,classes,templates,model)
    
    return zeroshot_weights, keys, petadata

def load_pa100k(model):
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    keys=["gender","age","body","accessory","carry","upperbody", "lowerbody", "foot"]
    data=pa100kbaseDataset(root_path)
    classes=data.classes
    templates=data.templates
    zeroshot_weights=text_classfier_weights(keys,classes,templates,model) 
    #zeroshot_weights=text_classfier_weights_all(keys,classes,templates,model)
    return zeroshot_weights, keys,data


def load_rap(model):
    keys=["head","age","gender","attach","action","foot","upperbody","lowerbody"]
    #root_path="/raid2/yue/datasets/Attribute-Recognition/"
    data=parbaseDataset()#(root_path)
    #pdb.set_trace()
    classes=data.classes
    templates=data.templates
    zeroshot_weights=text_classfier_weights(keys,classes,templates,model) 
    #zeroshot_weights=text_classfier_weights_all(keys,classes,templates,model)
    
    return zeroshot_weights, keys, data

def load_data(hparams,model):
    #pdb.set_trace()
    if hparams.testset=="PA100K":
        test_root="../dataset/PA-100K/PA100k_test/"
        zeroshot_weights, keys,data=load_pa100k(model)
        
    elif hparams.testset=="PA100KTrain":
        test_root="../dataset/PA-100K/PA100k_train_label/"
        zeroshot_weights, keys,data=load_pa100k(model)
       
    elif hparams.testset=="PETATrain":
        test_root="../dataset/PETAdata/PETA_train_label/"
        zeroshot_weights, keys,data=load_peta(model)
    
    elif hparams.testset=="RAPv1Train":
        test_root="../dataset/RAPv1/RAPv1_train/"
        zeroshot_weights, keys,data=load_rap(model)
    
    elif hparams.testset=="RAPv1":
        test_root="../dataset/RAPv1/RAPv1_test/"
        zeroshot_weights, keys,data=load_rap(model)

        
    else: #petatest
        test_root="../dataset/PETAdata/PETA_select_test/"
        zeroshot_weights, keys,data=load_peta(model)
       
    test_dataset=TextImageDataset(folder=test_root, image_size=hparams.imgSize, batch_size=hparams.minibatch_size,test=True)
    dataloader=DataLoader(dataset=test_dataset, batch_size=hparams.minibatch_size, shuffle=False)
    return dataloader,zeroshot_weights, keys,data

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))  
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def accuracy_t(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    pred2=torch.ones(len(pred))
    new_target=torch.ones(len(pred))
    for i in range(len(pred)):
        index=pred[i]
        new_target[i]=target[0][index]
    correct = pred2.eq(new_target) 
    #correct = pred.eq(target.view(1, -1).expand_as(pred))  
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def get_rank_images(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    new_target=torch.ones(len(pred))
    for i in range(len(pred)):
        index=pred[i]
        new_target[i]=target[0][index]
    return pred,new_target


def get_word(one_label,attr_words):
    caption=[]
    index=np.nonzero(one_label)
    for ite in index[0]:                       
        at=attr_words[ite]
        caption.append(at)
    return caption

def convert_labels(item,classes,caption,label,gtlabel):
    target=[]
    #print("item",item)
    #pdb.set_trace()
    cls=get_class(item,classes)
    if len(cls)==2:
        #pdb.set_trace()
        ind=gtlabel[item.split("_")[0]][cls[0]]                
        if(label[ind])==1:  #labels 里有标签的类别放在二分类的第一位
            tt=0
        else: 
            tt=1
        target.append(tt)    
    else:
        intar=set(cls)&set(caption)   
        if len(intar)>0:
            intar2=list(intar)
            for ith in intar2:     
                tt=cls.index(ith)
                target.append(tt)            

    return target


def T2I(hparams,dataloader,model,zeroshot_weights,data,keys): 
    classes=data.classes
    gtlabels=data.labels
    pred_logits=np.zeros((data.test_size,data.test_label_num))   #(90000,26) train
    labels=np.zeros((data.test_size,data.test_label_num)) 
    with torch.no_grad():
        logits_all={}
        t_num=0
        for i, (image_tensor, description, name, label) in enumerate(dataloader):                               
            if ("PA100K" in hparams.testset) or ("RAPv1" in hparams.testset):
                #pdb.set_trace()
                for la in label:
                    aa=la.split('[')[1].split(']')[0]
                    aa=list(aa.split(" "))

                    labels[t_num]=aa
                    t_num+=1      
            else:
                for la in label:
                    aa=ast.literal_eval(la)            
                    labels[t_num]=aa[:data.test_label_num]
                    t_num+=1
            image_tensor=image_tensor.cuda()
            image_features, attention = model.encode_image(image_tensor,k_num=8)
            k_value=get_k_value(hparams)
            for item in keys:
                ite=item.split("_")[0]
                k=int(k_value[ite])   
                #pdb.set_trace()
                image_features[:,k,:] /= image_features[:,k,:].norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * zeroshot_weights[item].t() @ image_features[:,k,:].t()

                logits=logits.cpu()
                if item in logits_all.keys():
                    logits_all[item]=torch.cat((logits_all[item],logits),1)
                else:
                    logits_all[item]=logits
          
    count,top1,top2={},{},{}
    rank_k_index, rank_k_labels,desc={},{},{}  
      
    for item in keys:
        cls=get_class(item,classes)
        for i in range(len(cls)):
            ite=cls[i]
            if ite in gtlabels[item.split("_")[0]].keys():
                index=gtlabels[item.split("_")[0]][ite]
                target=labels[:,index]
                #pdb.set_trace()
                input=logits_all[item][i:i+1,:]
                target=(torch.tensor(target)).unsqueeze(0)
                acc1, acc2 = accuracy_t(input, target, topk=(1,5))  #先搞定top1，然后再加if找top-5
                index_k_image,index_k_label=get_rank_images(input, target, topk=(1,5 ))
                top1[ite]=acc1  
                top2[ite]=acc2
                rank_k_index[ite]=index_k_image
                rank_k_labels[ite]= index_k_label
                desc[ite]=[data.templates[item.split("_")[0]][0].replace("{}",ite)]
    with open('PA100K_test_logits_T2I.pkl', 'wb') as f:
        pickle.dump({"rank_k_index": rank_k_index,"rank_k_labels":rank_k_labels,"desc":desc}, f)
    
    a1,a2=0,0
    t=0
    for item in top1.keys():
        if top1[item]>0:
            a1+=1
        if top2[item]>0:
            a2+=1
        t+=1
    a1=a1/t
    a2=a2/t
    #pdb.set_trace()
    print(top1,top2)
    print(a1,a2)

def multClass(pred_logits,labels):
    pred_logits=torch.tensor(pred_logits)
    labels=torch.tensor(labels)
    a1,a2=0,0
    pdb.set_trace()
    for i in range(len(labels)):
        indices = torch.nonzero(labels[i]==1, as_tuple=True)[0]
        
        for item in indices:
            acc1, acc2 = accuracy(pred_logits[i], item, topk=(1,5 ))
            a1+=acc1
            a2+=acc2
    pdb.set_trace()
    print(a1,a2)
    return a1,a2


                
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

def main(hparams):

    model_path="./lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt"     #PETA
    #model_path="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-14TScls_RAPv1/lightning_logs/version_6_best/epoch=83-step=21755.ckpt" #RAPv1
       
    #加载模型
    #pdb.set_trace()
    model=load_model(hparams,model_path)
    #加载数据
    dataloader,zeroshot_weights, keys,data=load_data(hparams,model)
    #测试数据  
    #get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys)
    T2I(hparams,dataloader,model,zeroshot_weights,data,keys)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--testset', type=str, required=True, help='[PA100K,PA100KTrain,PETA,PETATrain]')
    parser.add_argument('--imgSize', type=int, default=224, help='input image size')
    parser.add_argument('--trainset', type=str, required=True, help='[PA100K,PETA,RAPv1]')
    args = parser.parse_args()

    main(args)