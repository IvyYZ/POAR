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
from data.pre_rap1 import parbaseDataset
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
            k_value={"gender":0,"age":1,"body":2,"accessory":3,"carry":4,"upperbody":5, "lowerbody":6, "foot":7}  #pa100k
    elif "RAPv1" in hparams.testset: 
        if hparams.trainset=="PA100K":
            k_value={"head":2,"age":1,"gender":0,"attach":4,"action":3,"foot":7,"upperbody":5,"lowerbody":6,"id":2,"body":2,"accessory":3}  #pa100k--rapv1
        else:
            k_value={"head":0,"age":1,"gender":2,"attach":3,"action":4,"foot":5,"upperbody":6,"lowerbody":7,"id":2,"body":1,"accessory":0}
        # if hparams.trainset=="PA100K":
        #     k_value={"head":2,"age":1,"gender":0,"attach":4,"action":3,"foot":7,"upperbody":5,"lowerbody":6,"body":1,"id":1}  #pa100k--rapv1
        # else:
        #     k_value={"head":0,"age":1,"gender":2,"attach":3,"action":4,"foot":5,"upperbody":6,"lowerbody":7,"body":1,"id":1}
    else:       
        if hparams.trainset=="PA100K":
            k_value={"hair":2,"age":1,"gender":0,"carry":4,"accessory":3, "foot":7, "upperbody":5, "lowerbody":6} #pa100k->peta
        else:
            k_value={"hair":0,"age":1,"gender":2,"carry":3,"accessory":4, "foot":5, "upperbody":6, "lowerbody":7} #peta
    return k_value




def load_model(hparams):
     #加载模型
    print("Torch version:", torch.__version__)
    clip.available_models()    
    clp, _ = clip.load("ViT-B/16", device=device)
 

    model=clp.cuda()
    model.eval()
    
    return model

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
    #keys=["head","age","gender","attach","action","foot","upperbody","lowerbody"]
    keys=["head","age","gender","attach","action","foot","upperbody","lowerbody","body","id","accessory"]
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

def get_word(data, one_label,attr_words):
    caption=[]
    index=np.nonzero(one_label)
    for ite in index[0]:                       
        at=attr_words[ite]
        caption.append(at)
    target, describ= data.get_one_target(caption)
    return target

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
        intar=set(cls)&set(caption[item.split("_")[0]])   
        if len(intar)>0:
            intar2=list(intar)
            for ith in intar2:     
                tt=cls.index(ith)
                target.append(tt)            

    return target

def get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys): 
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
            image_features = model.visual(image_tensor.half())
            k_value=get_k_value(hparams)
            for item in keys:
                ite=item.split("_")[0]
                k=int(k_value[ite])   
                #pdb.set_trace()
                image_features/= image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * image_features @ zeroshot_weights[item]

                logits=logits.cpu()
                if item in logits_all.keys():
                    logits_all[item]=torch.cat((logits_all[item],logits),0)
                else:
                    logits_all[item]=logits
        
        with open('RAPv1_logits.pkl', 'wb') as f:
            pickle.dump({"logits_all": logits_all,"labels":labels}, f)
        
    count,top1,top2={},{},{}   
    for item in keys:
        count[item]=0
        top1[item]=0
        top2[item]=0
    for  i in range(len(labels)):
        #pdb.set_trace()
        caption=get_word(data,labels[i],data.attr_words)
        for item in keys:
            target=convert_labels(item,classes,caption,labels[i],gtlabels)#convert_labels(item,classes,caption)
            # if item =="body":
            #     pdb.set_trace()
            input=logits_all[item][i].unsqueeze(0).cuda()
            if len(target)>0:
                count[item]+=1
                t1,t2=0,0
                
                for tt in target:
                    tt=torch.tensor(tt).unsqueeze(0).cuda()
                    acc1, acc2 = accuracy(input, tt, topk=(1,2 ))  #先搞定top1，然后再加if找top-5
                    if t1<=acc1:
                        t1,t2=acc1,acc2
                top1[item]+=t1
                top2[item]+=t2
    
    print(top1,top2)
    accf,accf2={},{}  #每个clstoken 的准确率                          
    for item in keys:
        if count[item]>0:
            accf[item]=(top1[item]/count[item]) * 100
            accf2[item]=(top2[item]/count[item]) * 100
    print("Top_1,Top_2",accf,accf2)
    a1,a2,t=0,0,0 #整体准确率
    for item in accf.keys():
        a1+=accf[item]
        a2+=accf2[item]
        t+=1
    a1=a1/t
    a2=a2/t

    print("top1 and top2:",a1,a2)
    pdb.set_trace()



def T2I(hparams,dataloader,model,zeroshot_weights_all,data,keys): 
    classes=data.classes
    gtlabels=data.labels
    pred_logits=np.zeros((data.test_size,data.test_label_num))   #(90000,26) train
    labels=np.zeros((data.test_size,data.test_label_num)) 
    with torch.no_grad():
        logits_all={}
        t_num=0
        for i, (image_tensor, description, name, label) in enumerate(dataloader):                               
            if ("PA100K" in hparams.testset) or ("RAPv1" in hparams.testset):
                pdb.set_trace()
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
                logits = model.logit_scale.exp() * image_features[:,k,:] @ zeroshot_weights_all

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
    
    count,top1,top2={},{},{}
    pdb.set_trace()
    for item in keys:
        count[item]=0
        top1[item]=0
        top2[item]=0
    for  i in range(len(labels)):
         for item in keys:
            target=convert_labels(hparams,item,classes,gtlabels,labels)
            input=logits_all[item][i]
            if len(target)>0:
                count[item]+=1
                t1,t2=[],[]
                for tt in target:
                    tt=torch.tensor(tt).unsqueeze(0).cuda()
                    acc1, acc2 = accuracy(logits, tt, topk=(1,2 ))  #先搞定top1，然后再加if找top-5
                    t1.append(acc1)
                    t2.append(acc2)
                top1[item]+=max(t1)
                top2[item]+=max(t2)
    pdb.set_trace()
    print(top1,top2)

def Traverse_threshold(logits,labels,pred_labels):
    thres={}
    accuracy={}
    labels=np.array(labels)
    # with open('results.pkl', 'wb') as f:  
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
def convert_logits(hparams,keys,classes,gtlabels,logits_all,pred_logits):
    if "PA100K" in hparams.testset:
        for item in keys:
                print("item",item)
                kn=item.split("_")            
                ite=item.split("_")[0]
                _,index=torch.max((logits_all[item]),dim=1)  
                #pdb.set_trace()
                for i in range(len(index)):
                    id=index[i]
                    catg=classes[item][id]                                                          
                    if catg in gtlabels[item].keys():
                        index_t=gtlabels[item][catg]
                        pred_logits[i][index_t]=1     
        
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
            _,index=torch.max((logits_all[item]),dim=1)  
            for i in range(len(index)):  #代码还需要修改
                id=index[i]
                catg=classes[itemC][id]                                                          
                if catg in gtlabels[ite].keys():
                    index_t=gtlabels[ite][catg]
                    pred_logits[i][index_t]=1    

    return pred_logits

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

    #model_path="./lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt"     
    #model_path="./lightning_logs/version_3/checkpoints/epoch=14-step=5084.ckpt"   
    #model_path="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-14TScls/lightning_logs/version_17/epoch=10-step=3728.ckpt" 
    
    #加载模型
    #pdb.set_trace()
    model=load_model(hparams)
    #加载数据
    dataloader,zeroshot_weights, keys,data=load_data(hparams,model)
    print("load_train model:",hparams.trainset)
    print("load_test dataset:",hparams.testset)
    #测试数据  
    get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys)

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