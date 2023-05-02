from pydoc import describe
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import shutil

import os
import pdb
import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
#from sentence_transformers import SentenceTransformer
import pdb
import sys
sys.path.append("..")

np.random.seed(0)
random.seed(0)

os.environ['TORCH-HOME']='/raid2/yue/torch-model'

attr_words = [
    'female',
    'age over 60', 'age 18 to 60', 'age less 18',
    'front', 'side', 'back',
    'hat', 'glasses', 
    'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
    'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
    'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'
]


class pa100kbaseDataset(Dataset):
    def __init__(self, root_path):
        self.classes = {}
        self.templates={}
        self.ages={}
        self.labels={}
        self.root_path=root_path
        self.test_size=10000
        self.test_label_num=26

        #35 classes
        self.ages["age"] = ['age over 60', 'age 18 to 60', 'age less 18']
        self.classes["gender"] = ["female","male"]
        self.classes["age"] =["over sixty", "between eighteen and sixty", "less eighteen"]
        self.classes["body"] = ['front', 'side', 'back', 'other'] #,'other'
        self.classes["carry"] = [ "hand bag", 'shoulder bag', 'backpack', 'hold objects in front', 'other'] #, 'other'
        self.classes["accessory"] = [ "glasses", "hat",'other']
        self.classes["upperbody"] = ['short sleeve', 'long sleeve', 'stride', 'logo', 'plaid', 'splice', 'other']#,'other'
        self.classes["lowerbody"] = ['stripe', 'pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'other'] #,'other'
        self.classes["foot"] = ['boots','other'] # 
      
        self.labels["gender"] = {"female":0}
        self.labels["age"] ={"over sixty":1, "between eighteen and sixty":2, "less eighteen":3}
        self.labels["body"] = {'front':4, 'side':5, 'back':6,}
        self.labels["accessory"] = { "glasses":8,  "hat":7}
        self.labels["carry"] = { "hand bag":9, 'shoulder bag':10, 'backpack':11, 'hold objects in front':12, }
        #self.classes["upperbodys"] = ['short sleeve', 'long sleeve', 'stride', 'logo', 'plaid', 'splice',]
        #self.classes["lowerbodys"] = ["Casual","Formal","Trousers",  "Skirt",  "Shorts", "Strip", "Plaid", "Jeans"]
        self.labels["upperbody"] = {'short sleeve':13, 'long sleeve':14, 'stride':15, 'logo':16, 'plaid':17, 'splice':18,}
        self.labels["lowerbody"] = {'stripe':19, 'pattern':20, 'long coat':21, 'trousers':22, 'shorts':23, 'skirt and dress':24,}
        self.labels["foot"] = {'boots':25} # 


        self.attr_words = [
            'female',
            'over sixty', 'between eighteen and sixty', 'less eighteen',
            'front', 'side', 'back',
            'hat', 'glasses', 
            'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
            'short sleeve', 'long sleeve', 'stride', 'logo', 'plaid', 'plice',
            'stripe', 'pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'
        ]

        self.templates["gender"]=[
            'This person is {}.',
        ]
        self.templates["age"] = [
            'The age of this person is {} years old.', ]
        self.templates["body"] = [
            'This person is {}.',]
        self.templates["upperbody"] = [
            'This person is wearing {} in upper body.',
        ]
        self.templates["lowerbody"] = [
            'This person is wearing {} in lower body.',
        ]
        self.templates["foot"] = [
            'This person is wearing {} in foot.',
        ]
        self.templates["carry"] = [
            'This person is carrying {}.',
         ]

        self.templates["accessory"] = [
            'This person is accessorying {}.',
        ]
    
    def path2rest(self, path, captions, split, label):
        #name = path.split("/")[-1]
        image=cv2.imread(path)
        image=cv2.resize(image,(224,224))
        tensor=torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255.0
    
        return [tensor, captions, path, split]

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
        '''
        attr_words = [
        'female',
        'age over 60', 'age 18 to 60', 'age less 18',
        'front', 'side', 'back',
        'hat', 'glasses', 
        'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
        'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
        'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots']
        '''
        k3=["gender",  "age","body","accessory","carry","upperbody","lowerbody","foot"]

        cap=caption
        target={}
        lower,upper=[],[]
        describes=[]
       
        for i in range (len(cap)):
            # if cap[i]=="lower stripe" or cap[i]=="lower pattern":
            #     pdb.set_trace()
            if 'age' in cap[i]:
                #pdb.set_trace()
                target['age']=[self.classes['age'][self.ages['age'].index(cap[i])]]
            elif "upper" in cap[i]:               
                cc=cap[i].split(" ")[1]
                #pdb.set_trace()
                if len(cc)==0:
                    cc="other"                   
                upper.append(cc)  
            elif "lower" in cap[i]:               
                cc=cap[i].split(" ")[1]
                if len(cc)==0:
                    cc="other"  
                lower.append(cc)                                 
            else:
                for item in self.classes.keys():
                    
                    if cap[i] in self.classes[item]:
                        if item in target.keys():
                            target[item].append(cap[i])
                        else:
                            target[item]=[cap[i]]
        if len(upper)>0:
            for ith in upper:
                target["upperbody"].append(ith)
        if len(lower)>0:
            for ith in lower:
                target["lowerbody"].append(ith)   
        #pdb.set_trace()
        for item in k3:
            #pdb.set_trace()
            if item=="gender" and item not in target.keys():
                target[item]=[self.classes[item][0]]
            if item=="foot" and item not in target.keys():
                target[item]=[self.classes[item][1]]
            
            if item not in target.keys():
                target[item]=["other"]

            if item in target.keys():
                for tem in target[item]:
                        describes.append(self.templates[item][0].replace("{}",tem))
            describes.append(";")
        #pdb.set_trace()
        return target,describes

    #---------------------------pa100k--------
    def make_dir(self,path):
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

    # def get_label_embeds(self,labels):
    #     #pdb.set_trace()
    #     model = SentenceTransformer('all-mpnet-base-v2')
    #     embeddings = model.encode(labels)
    #     return embeddings

    def get_cls(self):
        
        cls_dic={}
        tokenized_text={}
        for item in self.classes.keys():
            clas_des=[]
            for ite in self.classes[item]:
                clas_des.append(self.templates[item][0].replace("{}",ite).split(".")[0])
            cls_dic[item]=clas_des
            from clip.clip import tokenize as tokenize
            tokenized_text[item] =tokenize(clas_des, truncate=True).tolist() 
        #pdb.set_trace()
        #np.savetxt("cls_35.txt",tokenized_text)

        f=open("cls_pa100k.txt","a")
        f.write(str(tokenized_text))
        f.close()
            
        return tokenized_text
    def generate_data_description(self,save_dir,root_path,save_root=None,phase=None):
        """
        create a dataset description file, which consists of images, labels
        """
        #pdb.set_trace()
        #pa100k_data = loadmat(os.path.join(save_dir, 'annotation','annotation.mat'))
        pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

        dataset = EasyDict()
        dataset.description = 'pa100k'
        #pdb.set_trace() 
        #dataset.root = os.path.join(save_dir, 'data','release_data','release_data')
        dataset.root = os.path.join(save_dir, 'data')

        train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
        val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
        test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
        dataset.image_name = train_image_name + val_image_name + test_image_name

        dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
        assert dataset.label.shape == (100000, 26)
        dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]

          
        dataset.attr_words = np.array(attr_words)
        #dataset.attr_vectors = self.get_label_embeds(attr_words)

        dataset.partition = EasyDict()
        dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
        dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
        dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
        dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    
        #pdb.set_trace()
        bs=[] 
        if phase=="test":          
            for i in range(len(dataset.partition.test)):
                key='test'
                for item in attr_words:
                    imagename=os.path.join(root_path, test_image_name[i])                 
                    index=np.nonzero(pa100k_data['test_label'][i])
                    #pdb.set_trace()
                    caption=[]
                    for ite in index[0]:                       
                        at=dataset.attr_words[ite]
                        caption.append(at)
                    target, captions = self.get_one_target(caption)               
                    b0=self.path2rest(imagename, target, key,dataset.label[i])
                    bs.append(b0)
        else:
            #pdb.set_trace()
            for i in range(len(dataset.partition.trainval)):
                key='trainval'
                
                imagename=os.path.join(root_path,'release_data','release_data', dataset.image_name[i])                 
                index=np.nonzero(dataset.label[i])
                #pdb.set_trace()
                caption=[]
                for ite in index[0]:                       
                    at=dataset.attr_words[ite]
                    caption.append(at)
                target, captions = self.get_one_target(caption)               
                b0=self.path2rest(imagename, target, key,dataset.label[i])
                #bs.append(b0)
                if key !="test":
                    if save_root!=None:
                        #pdb.set_trace()
                        f=open(os.path.join(save_root, dataset.image_name[i].split(".")[0]+".txt"), 'a') 
                        for tt in captions:
                            f.write(tt)
                        f.close() 
                        shutil.copy(imagename, os.path.join(save_root,dataset.image_name[i])) 
                    #b0=self.path2rest(imagename, target, key,label)
                    #bs.append(b0)
                    
       
        return bs



        # dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
        # dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

        # with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        #     pickle.dump(dataset, f)
    

    




if __name__ == '__main__':

    root_path="./PA-100K/data/"
    save_dir = './PA-100K/'
    save_root="./PA100K/"
    # keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    
    # if os.path.exists(save_root):
    #     shutil.rmtree(save_root)
    #     os.makedirs(save_root)
    # else:
    #     os.makedirs(save_root)
   
    #加载数据
    petadata=pa100kbaseDataset(root_path)
    classes=petadata.classes
    templates=petadata.templates
    #bs=petadata.read_mat(root_path, save_root,phase="train")


    petadata.get_cls()
    #bs=petadata.generate_data_description(save_dir,root_path,save_root,phase="train")
