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
    #["["gender"] = ["female","male"]["body"]=['fat', 'normal', 'thin']["id"]=['customer', 'clerk'] ["head"]=[ 'black hair','long hair', 'bald head',]"accessory['hat', 'glasses', 'muffler',"other"]
    #pdb.set_trace()
    if "PA100K" in hparams.testset:  
        if hparams.trainset=="PETA":
            k_value={"gender":2,"age":1,"body":2,"accessory":4,"carry":3,"upperbody":6, "lowerbody":7, "foot":5}  #peta->pa100k
        else:
            k_value={"gender":2,"age":1,"body":2,"accessory":4,"carry":3,"upperbody":6, "lowerbody":7, "foot":5} 
            #k_value={"gender":0,"age":1,"body":2,"accessory":3,"carry":4,"upperbody":5, "lowerbody":6, "foot":7}  #pa100k
    elif "RAPv1" in hparams.testset: 
        if hparams.trainset=="PA100K":
            k_value={"head":2,"age":1,"gender":0,"attach":4,"action":3,"foot":7,"upperbody":5,"lowerbody":6,"id":2,"body":2,"accessory":3}  #pa100k--rapv1
        else:
            k_value={"head":0,"age":1,"gender":2,"attach":3,"action":4,"foot":5,"upperbody":6,"lowerbody":7,"id":2,"body":1,"accessory":0}
    else:       
        if hparams.trainset=="PA100K":
            k_value={"hair":2,"age":1,"gender":0,"carry":4,"accessory":3, "foot":7, "upperbody":5, "lowerbody":6} #pa100k->peta
        else:
            k_value={"hair":0,"age":1,"gender":2,"carry":3,"accessory":4, "foot":5, "upperbody":6, "lowerbody":7} #peta
    return k_value


def load_peta(model):
    keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    petadata=petabaseDataset(root_path)    
    return keys, petadata

def load_pa100k(model):
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    keys=["gender","age","body","accessory","carry","upperbody", "lowerbody", "foot"]
    data=pa100kbaseDataset(root_path)
    
    return keys,data


def load_rap(model):
    keys=["head","age","gender","attach","action","foot","upperbody","lowerbody","id","body","accessory"]
    #root_path="/raid2/yue/datasets/Attribute-Recognition/"
    data=parbaseDataset()#(root_path)
       
    return keys, data

def load_data(hparams,model):
    #pdb.set_trace()
    if hparams.testset=="PA100K":
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/PA-100K/PA100k_test/"
        #train_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/PA-100K/PA100k_train_label/"
        keys,data=load_pa100k(model)
        
    elif hparams.testset=="PA100KTrain":
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/PA-100K/PA100k_train_label/"
        keys,data=load_pa100k(model)
       
    elif hparams.testset=="PETATrain":
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/PETAdata/PETA_train_label/"
        keys,data=load_peta(model)
    
    elif hparams.testset=="RAPv1Train":
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/RAPv1/RAPv1_train/"
        keys,data=load_rap(model)
    
    elif hparams.testset=="RAPv1":
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/RAPv1/RAPv1_test/"
        keys,data=load_rap(model)

        
    else: #petatest
        test_root="/home/xiaodui/Dataset/all_dataset/PAR/rebuilt.dataseta/dataset/PETAdata/PETA_select_test/"
        keys,data=load_peta(model)
       
    test_dataset=TextImageDataset(folder=test_root, image_size=hparams.imgSize, batch_size=hparams.minibatch_size,test=True)
    dataloader=DataLoader(dataset=test_dataset, batch_size=hparams.minibatch_size, shuffle=False)
    return dataloader,keys,data


def main(hparams):
    #load data
    dataloader,zeroshot_weights, keys,data=load_data(hparams,model)
    print("load_train model:",hparams.trainset)
    print("load_test dataset:",hparams.testset)
    #test data
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