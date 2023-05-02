import ast
from typing_extensions import Self
import numpy as np
import torch
from tqdm import tqdm
from test.classifierWeights import text_classfier_weights_vtb,get_pedestrian_metrics,get_pedestrian_metrics0
from data.pre_cls_pa100k2 import pa100kbaseDataset
from data.pre_peta_random import petabaseDataset
from data.pre_rap1 import parbaseDataset
import pdb
import copy
from pytorch_lightning import Trainer
from argparse import ArgumentParser
#from data.text_image_dm import TextImageDataModule,TextImageDataset
from torch.utils.data import Dataset,DataLoader
import ast
from VTB.models.base_block import *
from sentence_transformers import SentenceTransformer
import os

os.environ['TORCH-HOME']='/raid2/yue/torch-model'

import pickle
import torch.nn.functional as F
import PIL

os.environ['TORCH-HOME']='/raid2/yue/torch-model'
device = "cuda" if torch.cuda.is_available() else "cpu"


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from random import randint, choice
import pdb



class TextImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 image_size=224,
                 batch_size=32,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=False
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = sorted(list(keys))
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(self.fix_img),
            # T.RandomResizedCrop(image_size,
            #                     scale=(self.resize_ratio, 1.),
            #                     ratio=(1., 1.)),
            T.Resize(size=(232, 232)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            T.RandomErasing(p=0.5, scale=(0.03, 0.08), ratio=(0.3, 3.3)),
        ])

        self.image_test_transform = T.Compose([
            T.Lambda(self.fix_img),
            T.Resize(size=(256, 128)), 
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),           
        ])

        self.custom_tokenizer = custom_tokenizer
        self.labels=[]
        self.batch_size=batch_size

    def __len__(self):
        return len(self.keys)
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


    def __getitem__(self, ind):
        #print("ind",ind)
        key = self.keys[ind]

        text_file = self.text_files[key]
        descriptions = text_file.read_text().split(';')
        description = list( descriptions)[:8]
        labels=list( descriptions)[8]
       
        image_file = self.image_files[key]
        image = PIL.Image.open(image_file).convert("RGB")
        
           
        if len(labels)>0:
            image_tensor=self.image_test_transform(image)
            return  image_tensor, description, key, labels
        else:
            image_tensor = self.image_transform(image)
            return  image_tensor, description, key
thres={}

def load_model(hparams,model_path):
     #加载模型
    attr_num=26 #pa100k
    model = TransformerClassifier(attr_num)
    
    #pdb.set_trace()
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dicts'],strict=False)
    model2=model.cuda()
    model2.eval()
    
    return model2

def load_peta(model):
    keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    petadata=petabaseDataset(root_path)
    classes=petadata.classes
    templates=petadata.templates
     
    zeroshot_weights=text_classfier_weights_vtb(keys,classes,templates,model)
    
    return zeroshot_weights, keys, petadata

def load_pa100k(model):
    root_path="/raid2/yue/datasets/Attribute-Recognition/PA-100K/data/"
    keys=["gender","age","body","accessory","carry","upperbody", "lowerbody", "foot"]
    data=pa100kbaseDataset(root_path)
    classes=data.classes
    templates=data.templates
    zeroshot_weights=text_classfier_weights_vtb(keys,classes,templates,model)
    return zeroshot_weights, keys,data

def load_rap(model):
    keys=["head","age","gender","attach","action","foot","upperbody","lowerbody","id","body","accessory"]
    #root_path="/raid2/yue/datasets/Attribute-Recognition/"
    data=parbaseDataset()#(root_path)
    #pdb.set_trace()
    classes=data.classes
    templates=data.templates
    zeroshot_weights=text_classfier_weights_vtb(keys,classes,templates,model) 
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
       
    test_dataset=TextImageDataset(folder=test_root, image_size=hparams.imgSize, batch_size=hparams.minibatch_size)
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
    cls=get_class(item,classes) #classes: gt
    
    if len(cls)==2:
        ind=gtlabel[item.split("_")[0]][cls[0]]                
        if(label[ind])==1:  #labels 里有标签的类别放在二分类的第一位
            tt=0
        else: 
            tt=1
        target.append(tt)    
    else:
        try:
            intar=set(cls)&set(caption[item.split("_")[0]])   #caption: 当前图像的类别
            if len(intar)>0:
                intar2=list(intar)
                for ith in intar2:     
                    tt=cls.index(ith)
                    target.append(tt) 
        except:
            pdb.set_trace()          

    return target

def get_images_item(hparams,dataloader,model,zeroshot_weights,data,keys): 
    classes=data.classes
    gtlabels=data.labels
    pred_logits=np.zeros((data.test_size,data.test_label_num))   #(90000,26) train
    labels=np.zeros((data.test_size,data.test_label_num)) 
    model2 = SentenceTransformer('all-mpnet-base-v2')
    with torch.no_grad():
        logits_all={}
        t_num=0
        texts = model2.encode(data.attr_words)
        texts=torch.tensor(texts)
        texts=texts.cuda()
        for i, (image_tensor, description, name, label) in enumerate(dataloader):                               
            #imgs=preprocess(Image.open(imgpath+name).convert('RGB'))
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
            #pdb.set_trace()

            logits,image_features = model(image_tensor,texts)
            #x2 = model.vit(image_tensor)
            image_features[:,0,:] /= image_features[:,0,:].norm(dim=-1, keepdim=True)
            #pdb.set_trace()
            for item in keys:                
                ite=item.split("_")[0]               
                logits = 100* image_features[:,0,:] @ zeroshot_weights[item]
                logits=logits.cpu()
                if item in logits_all.keys():
                    logits_all[item]=torch.cat((logits_all[item],logits),0)
                else:
                    logits_all[item]=logits
       # pred_logits2=copy.deepcopy(pred_logits)   
        #pred_logits=convert_logits_top1(hparams,keys,classes,gtlabels,logits_all,pred_logits)      #convert_logits_top1
        #pred_logits2=convert_logits_ma(hparams,keys,classes,gtlabels,logits_all,pred_logits2)

    #pred_labels=copy.deepcopy(pred_logits)
    #acc=Traverse_threshold(pred_logits2,labels,pred_labels)

    count,top1,top2={},{},{}   
    for item in keys:
        count[item]=0
        top1[item]=0
        top2[item]=0
    for  i in range(len(labels)):
        #pdb.set_trace()
        caption=get_word(data,labels[i],data.attr_words)
        for item in keys:
            target=convert_labels(item,classes,caption,labels[i],gtlabels) #convert_labels(item,classes,caption)
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

def convert_logits_top1(hparams,keys,classes,gtlabels,logits_all,pred_logits):
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
                
def convert_logits_ma(hparams,keys,classes,gtlabels,logits_all,pred_logits):
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
    # with open('results.pkl', 'wb') as f:  
    #     pickle.dump({"logits": logits,"labels":labels}, f)
    #pdb.set_trace()
    for i in range(len(logits[0])):
        print(i,"-th col start search----")
        sort_list=np.sort(logits[:,i])
        d1=labels[:,i:i+1]  
        d2=logits[:,i:i+1]
        a1=0
        a2=0
        t1=0
        for thre in sort_list:              
            a3,_=get_pedestrian_metrics(d1, d2,threshold=thre)
            if a3.label_acc>a1:            #label_acc,add_acc 
                a1=a3.label_acc 
                a2=copy.deepcopy(a3)
                t1=thre
        print("thres[",i,"] is:", t1)
        #print("the best label_acc is:",a2.label_acc)
        thres[i]=t1
        accuracy[i]=a2
        a3,pred_label_best=get_pedestrian_metrics(d1, d2,threshold=t1)
        pred_labels[:,i:i+1]=pred_label_best
 
    acc=get_pedestrian_metrics0(labels, pred_labels)  
    print(acc)     
    pdb.set_trace()
    return acc

def main(hparams):

    model_path="/raid2/yue/ReID/vision_language/VTB/VTB2/logs/PA100k/ckpt_2022-07-07_23_30_38_19.pth"
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
    #parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    parser.add_argument('--testset', type=str, required=True, help='[PA100K,PA100KTrain,PETA,PETATrain]')
    parser.add_argument('--trainset', type=str, required=True, help='[PA100K,PETA]')
    parser.add_argument('--imgSize', type=int, default=224, help='input image size')
    args = parser.parse_args()

    main(args)