import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
import cv2
import shutil
from torch.utils.data.dataset import Dataset
import pdb
#from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)
'''
['Female', 'AgeLess16', 'Age17-30', 'Age31-45', 
'BodyFat', 'BodyNormal', 'BodyThin', 'Customer', 'Clerk', 
'hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses', 'hs-Muffler', 
'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 
'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers', 
'shoes-Leather', 'shoes-Sport', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 
'attach-Backpack', 'attach-SingleShoulderBag', 'attach-HandBag', 'attach-Box', 'attach-PlasticBag', 'attach-PaperBag', 'attach-HandTrunk', 'attach-Other', 
'action-Calling', 'action-Talking', 'action-Gathering', 'action-Holding', 'action-Pusing', 'action-Pulling', 'action-CarrybyArm', 'action-CarrybyHand']
'''


class parbaseDataset(Dataset):
    def __init__(self):
        self.classes = {}
        self.templates={}
        self.labels={}
        self.ages={}
        #self.root_path=root_path
        self.test_size=8317 #33268,33255,8317
        self.test_label_num=51

        #51 classes
        self.ages["age"] = ['age less 16', 'age 17 30', 'age 31 45',]
        self.classes["gender"] = ["female","male"]
        self.classes["body"]=['fat', 'normal', 'thin']
        self.classes["id"]=['customer', 'clerk']
        self.classes["age"] =["less seventeen","between seventeen and thirty", "between thirty-one and forty-five",]
        self.classes["head"]=[ 'black hair','long hair', 'bald head',"other"]
        self.classes["accessory"]=['hat', 'glasses', 'muffler','other']
        #self.classes["body"] = ['fat', 'normal', 'thin', 'customer', 'clerk']
        self.classes["attach"] = [ 'backpack', 'shoulder bag', 'hand bag', 'box', 'plastic bag', 'paper bag',
    'hand trunk', 'other',"nothing"] #
        self.classes["action"] = [ 'calling', 'talking', 'gathering', 'holding', 'pushing', 'pulling',
    'carry arm', 'carry hand',"nothing special"] 
        self.classes["upperbody"] = ['shirt', 'sweater', 'vest', 't-shirt', 'cotton', 'jacket', 'suit up','tight', 'short sleeve','other']
        self.classes["lowerbody"] = ['long trousers', 'skirt', 'short skirt', 'dress', 'jeans', 'tight trousers',"other"]
        self.classes["foot"] = ['leather', 'sport', 'boots', 'cloth', 'casual','other'] # 
        self.attr_words = [
        'female', 'age less 16', 'age 17 30', 'age 31 45',
        'body fat', 'body normal', 'body thin', 'customer', 'clerk',
        'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler',
        'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
        'upper tight', 'upper short sleeve',
        'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',
        'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',
        'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
        'attach hand trunk', 'attach other',
        'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
        'action carry arm', 'action carry hand']

 

        self.labels["age"] = {'age less 16', 'age 17 30', 'age 31 45',}
        self.labels["gender"] = {"female":0,}
        self.labels["body"]={'fat':4, 'normal':5, 'thin':6, }
        self.labels["id"]={'customer':7, 'clerk':8}
        self.labels["age"] ={"less seventeen":1,"between seventeen and thirty":2, "between thirty-one and forty-five":3}
        self.labels["head"]={'bald head':9, 'long hair':10, 'black hair':11, }
        self.labels["accessory"]={'hat':12, 'glasses':13, 'muffler':14,}
        self.labels["attach"] = {'backpack':35, 'shoulder bag':36, 'hand bag':37, 'box':38, 'plastic bag':39, 'paper bag':40,
    'hand trunk':41, 'other':42}
        self.labels["action"] ={'calling':43, 'talking':44, 'gathering':45, 'holding':46, 'pushing':47, 'pulling':48,
    'carry arm':49, 'carry hand':50}
        self.labels["upperbody"] = {'shirt':15, 'sweater':16, 'vest':17, 't-shirt':18, 'cotton':19, 'jacket':20, 'suit up':21,'tight':22, 'short sleeve':23}
        self.labels["lowerbody"] = {'long trousers':24, 'skirt':25, 'short skirt':26, 'dress':27, 'jeans':28, 'tight trousers':29,}
        self.labels["foot"] = {'leather':30, 'sport':31, 'boots':32, 'cloth':33, 'casual':34} # 

        self.templates["gender"]=[
            'This person is {}.',
        ]
        self.templates["body"]=[
            'This person is {}.',
        ]
        self.templates["id"]=[
            'This person is {}.',
        ]
        self.templates["accessory"] = [
            'This person is accessory {}.',
        ]
        self.templates["age"] = [
            'The age of this person is {} years old.', ]
        self.templates["head"] = [
            'This person has {}.',]
        self.templates["upperbody"] = [
            'This person is wearing {} in upper body.',
        ]
        self.templates["lowerbody"] = [
            'This person is wearing {} in lower body.',
        ]
        self.templates["foot"] = [
            'This person is wearing {} in foot.',
        ]
        self.templates["attach"] = [
            'This person is carrying {}.',
         ]

        self.templates["action"] = [
            'This person is {}.',
        ]
    
    def path2rest(self, path, captions, split, label):
        #name = path.split("/")[-1]
        image=cv2.imread(path)
        image=cv2.resize(image,(224,224))
        tensor=torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255.0
    
        return [tensor, captions, path, split,label]

    def get_id_names(self, filenames):
        id=[]
        for i in range (len(filenames)):
            id.append(filenames[i].split("_")[0])
        return id

    def get_id_file(self,lines):
        id=[]
        for i in range (len(lines)):
            id.append(lines[i].split(" ")[0])
        return id

    def get_one_target(self,caption):
        #k3=["gender",  "age","head","action","attach","upperbody","lowerbody","foot"]
        #k_value={"hair":0,"age":1,"gender":2,"carry":3,"accessory":4, "foot":5, "upperbody":6, "lowerbody":7}
        '''
           #     self.attr_words = [
    # 'female', 'age less 16', 'age 17 30', 'age 31 45',[0-3]
    # 'body fat', 'body normal', 'body thin', 'customer', 'clerk',[4-8]
    # 'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler', 9-14
    # 'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    # 'upper tight', 'upper short sleeve', [15-23]
    # 'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',[24-29]
    # 'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',[30-34]
    # 'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    # 'attach hand trunk', 'attach other',[35-42]
    # 'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    # 'action carry arm', 'action carry hand'][43-50]
        '''
        self.k3=["head","age","gender","attach","action","foot","upperbody","lowerbody","body","id"]

        cap=caption
        target={}
        lower,upper=[],[]
        describes=[]
       
        for i in range (len(cap)):
            if 'age' in cap[i]:
                target['age']=[self.classes['age'][self.ages['age'].index(cap[i])]]
            
  
            for item in self.classes.keys():  
                #pdb.set_trace()  
                ct=cap[i].split(" ")
                if len(ct)>1:
                    if len(ct)==2:
                        cc=ct[1]
                    else:
                        cc=ct[1]+" "+ct[2]
                else:
                    cc=ct[0]                   
                if cc in self.classes[item]:                   
                    if item in target.keys():
                        target[item].append(cc)
                    else:
                        target[item]=[cc]

        if "female" not in cap:
            if "gender" in target.keys():
                target["gender"].append("male")
            else:
                target["gender"]=["male"]
        
        if "attach" not in target.keys():
            target["attach"]=["nothing"]

        if "action" not in target.keys():
            target["action"]=["nothing special"]
        if "foot" not in target.keys():
            target["foot"]=["other"]
        if "head" not in target.keys():
            target["head"]=["other"]
        if "lowerbody" not in target.keys():
            target["lowerbody"]=["other"]
        
        if "upperbody" not in target.keys():
            target["upperbody"]=["other"]
        if "body" not in target.keys():
            target["body"]=["other"]
        
        if "accessory" not in target.keys():
            target["accessory"]=["other"]
 
        for item in self.k3:
            if item in target.keys():
                try:
                    for tem in target[item]:
                        describes.append(self.templates[item][0].replace("{}",tem))
                except:
                    pdb.set_trace()
            describes.append(";")
        #pdb.set_trace()
        return target,describes

    #---------------------------pa100k--------
    def make_dir(self,path):
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

    def generate_data_description(self,save_dir,save_root=None,phase=None):
        """
        create a dataset description file, which consists of images, labels
        """

        data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))

        dataset = EasyDict()
        dataset.description = 'rap'
        dataset.root = os.path.join(save_dir, 'RAP_dataset/')
        dataset.image_name = [data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
        raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
        # (41585, 92)
        raw_label = data['RAP_annotation'][0][0][1]
        dataset.label = raw_label[:, np.array(range(51))]
        dataset.attr_name = [raw_attr_name[i] for i in range(51)]

        dataset.attr_words = np.array(self.attr_words)
        #dataset.attr_vectors = get_label_embeds(attr_words)

        dataset.partition = EasyDict()
        dataset.partition.trainval = []
        dataset.partition.test = []

        dataset.weight_trainval = []

        for idx in range(5):
            #RAP_annotation.partion is the 5 random partion for training and test, which is the same as the setting in our paper.
            trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
            test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1

            dataset.partition.trainval.append(trainval)
            dataset.partition.test.append(test)

        #pdb.set_trace()
        bs=[] 
        if phase=="test":          
            for ith in dataset.partition.test[0]:
                key='test'
                imagename=os.path.join(save_dir, 'RAP_dataset/', dataset.image_name[ith])                      
                index=np.nonzero(dataset.label[ith])
                #pdb.set_trace()
                caption=[]
                for ite in index[0]:                       
                    at=dataset.attr_words[ite]
                    caption.append(at)
                target, captions = self.get_one_target(caption)
                f=open(os.path.join(save_root, dataset.image_name[ith].split(".")[0]+".txt"), 'a') 
                for tt in captions:
                    f.write(tt)
                f.write(str(dataset.label[ith])+";")
                f.close() 
                shutil.copy(imagename, os.path.join(save_root,dataset.image_name[ith]))                
                    # b0=self.path2rest(imagename, target, key,dataset.label[i])
                    # bs.append(b0)
        else:
            #pdb.set_trace()
            for ith in dataset.partition.trainval[0]:
                key='trainval'
                
                imagename=os.path.join(save_dir,'RAPv1_dataset/RAP_dataset/', dataset.image_name[ith])                
                index=np.nonzero(dataset.label[ith])
                #pdb.set_trace()
                caption=[]
                for ite in index[0]:                       
                    at=dataset.attr_words[ite]
                    caption.append(at)
                target, captions = self.get_one_target(caption)               
                #b0=self.path2rest(imagename, target, key,dataset.label[i])
                #bs.append(b0)
                if key !="test":
                    if save_root!=None:
                        #pdb.set_trace()
                        try:
                            f=open(os.path.join(save_root, dataset.image_name[ith].split(".")[0]+".txt"), 'a') 
                            for tt in captions:
                                f.write(tt)
                            f.write(str(dataset.label[ith])+";")
                            f.close() 
                            shutil.copy(imagename, os.path.join(save_root,dataset.image_name[ith])) 
                        except:
                            print(imagename, "is not exist!")



if __name__ == "__main__":
    save_dir = '/raid2/yue/datasets/Attribute-Recognition/RAP/RAPv1/'
    #save_root="./PA100K/"
    save_root="/raid2/yue/ReID/vision_language/train-CLIP-2th/dataset/RAPv1/RAPv1_train/"
    # keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
        os.makedirs(save_root)
    else:
        os.makedirs(save_root)
   
    #加载数据
    data=parbaseDataset()
    classes=data.classes
    templates=data.templates
    
    bs=data.generate_data_description(save_dir,save_root=save_root,phase="train")

