from pydoc import describe
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import shutil

import os
import pdb


class petabaseDataset(Dataset):
    def __init__(self, root_path):
        self.classes = {}
        self.templates={}
        self.ages={}
        self.labels={}
        self.root_path=root_path
        self.test_size=7600 #11400 #7600
        self.test_label_num=35

        #35 classes
        self.ages["age"] = ["Less15", "Less30", "Less45", "Less60", "Larger60"]
        self.classes["gender"] = ["Male","Female"]
        self.classes["age"] =["less fifteen", "between fifteen and thirty", "between thirty and forty-five", "between forty-five and sixty", "Larger sixty"]
        #["less fifteen", "less thirty", "less forty-five", "less sixty", "Larger sixty"]
        self.classes["hair"] = ["Long","Short"]
        self.classes["carry"] = [ "Backpack",   "MessengerBag","PlasticBags", "Other", "Nothing"]

        self.classes["accessory"] = [ "Sunglasses",  "Hat",  "Muffler", "Nothing"]
        self.classes["upperbodys"] = ["Casual","Formal","Jacket", "Logo",  "ShortSleeve", "Plaid", "ThinStripes","Tshirt", "VNeck","Other",]
        self.classes["lowerbodys"] = ["Casual","Formal","Trousers",  "ShortSkirt",  "Shorts",  "Plaid", "Jeans"]
        self.classes["upperbody"] = ["Jacket", "Logo",  "ShortSleeve", "Plaid", "ThinStripes","Tshirt", "VNeck","Other"]
        self.classes["lowerbody"] = ["Trousers",  "ShortSkirt",  "Shorts",  "Plaid", "Jeans"]
        self.classes["foot"] = ["LeatherShoes","Sandals", "Sneaker",'Shoes'] # 

        self.classes["style"] = ["Casual", "Formal"]
        self.classes["color"] = ["Black", "Blue", "Brown", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow","Green"]


        self.labels["age"]={"between fifteen and thirty":0, "between thirty and forty-five":1, "between forty-five and sixty":2, "Larger sixty":3,}
        self.labels["carry"]={'Backpack':4, 'Other':5, 'Nothing':20,'PlasticBags':22, 'MessengerBag':17, }
        self.labels["lowerbody"]={ 'Casual':6,'Formal':8, 'Jeans':12, 'Shorts':25, 'ShortSkirt':27, 'Trousers':31,}
        self.labels["upperbody"]={'Casual':7,  'Formal':9,  'Jacket':11,'Logo':14, 'Plaid':21, 'ShortSleeve':26,  'ThinStripes':29,  'Tshirt':32, 'Other':33, 'VNeck':34,}
        self.labels["hair"]={ 'Long':15, }
        self.labels["accessory"]={'Hat':10,'Muffler':18, 'Nothing':19, 'Sunglasses':30,}
        self.labels["foot"]={ 'LeatherShoes':13,'Sandals':23, 'Shoes':24,  'Sneaker':28,  }
        self.labels["gender"]={'Male':16,  }

        self.attr=["between fifteen and thirty", "between thirty and forty-five", 
        "between forty-five and sixty", "Larger sixty",'Backpack', 'Other',
       'Casual', 'Casual','Formal','Formal','Hat','Jacket','Jeans','LeatherShoes',
       'Logo','Long','Male','MessengerBag', 'Muffler', 
       'Nothing','Nothing','Plaid','PlasticBags','Sandals', 'Shoes',
       'Shorts','ShortSleeve','ShortSkirt','Sneaker','ThinStripes',
       'Sunglasses','Trousers', 'Tshirt','Other', 'VNeck',]

        self.attr_words=['personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBackpack', 
        'carryingOther', 'lowerBodyCasual', 'upperBodyCasual', 'lowerBodyFormal', 'upperBodyFormal', 
        'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans', 'footwearLeatherShoes', 'upperBodyLogo', 
        'hairLong', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 
        'carryingNothing', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 
        'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneaker', 
        'upperBodyThinStripes', 'accessorySunglasses', 'lowerBodyTrousers', 'upperBodyTshirt', 
        'upperBodyOther', 'upperBodyVNeck']


       # define_templates
        self.templates["35"]=[
            'This person has {}.',
            'This person is {}.',
            'This person is wearing {}.',
        ]

        self.templates["gender"]=[
            'This person is {}.',
        ]
        self.templates["age"] = [
            'The age of this person is {} years old.', ]
        self.templates["hair"] = [
            'This person has {} hair.',]
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
    
    def path2rest(self, path, captions, split,label):
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
        cap=caption
        target={}
        age,carry,accessory,lower,upper,foot,hair,cc,gender=[],[],[],[],[],[],[],[],[]
        describes=[]
        t1=""
        t2=""
       
        for i in range (len(cap)):
            if "personalMale" in cap [i] or "personalFemale" in cap [i]:
                gender.append(cap[i].split("personal")[1])
                #cont=self.templates["gender"][0].replace("{}",cap[i].split("personal")[1])
            elif "personal" in cap[i]:
                #pdb.set_trace()
                ag=cap[i].split("personal")[1]
                cot=self.classes["age"][self.ages["age"].index(ag)]
                age.append(cot)
            
            elif "carrying" in cap[i]:
                cc=cap[i].split("carrying")[1]
                if cc in self.classes["carry"]:
                    carry.append(cc)
               
            elif "accessory" in cap[i]:
                cc=(cap[i].split("accessory")[1]).split("\n")[0]
                if cc in self.classes["accessory"]:
                    accessory.append(cc)
               
                             
            elif "lower" in cap[i]: 
                cc=cap[i].split("lowerBody")[1] 
                if cc in  self.classes["lowerbodys"]: 
                    # if "Strip" in cc:
                    #     lower.append("Strip")
                    lower.append(cc)
                         
                    
               
            elif "upper" in cap[i]:
                cc=cap[i].split("upperBody")[1]
                if cc in self.classes["upperbodys"]:  
                    # if "Strip" in cc:
                    #     upper.append("Strip")
                    # else:
                    upper.append(cc)
               
            elif "foot" in cap[i]:
                cc=cap[i].split("wear")[1]
                if cc in self.classes["foot"]:
                    foot.append(cc)
               
            elif "hair" in cap[i]:
                #hair+=wa.split(cap[i].split("hair")[1])
                cc=cap[i].split("hair")[1]
                if cc in self.classes["hair"]:
                    hair.append(cc)                               
            else:
                cc.append(cap[i])


        target["hair"]=hair
        target["age"]=age   
        target["gender"]=gender

        target["carry"]=carry
        target["accessory"]=accessory

        target["upperbody"]=upper
        target["lowerbody"]=lower    
        target["foot"]=foot

        k3=["hair", "age","gender", "carry","accessory","foot","upperbody","lowerbody"]
        
        for item in k3:
            #print(item)
            for tem in target[item]:
                    describes.append(self.templates[item][0].replace("{}",tem))
            describes.append(";")
       
        return target, describes

    def read_mat(self, root_path,save_root=None,phase=None):
        '''
        dataset['att_name']
        ['personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBackpack', 
        'carryingOther', 'lowerBodyCasual', 'upperBodyCasual', 'lowerBodyFormal', 'upperBodyFormal', 
        'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans', 'footwearLeatherShoes', 'upperBodyLogo', 
        'hairLong', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 
        'carryingNothing', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 
        'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneaker', 
        'upperBodyThinStripes', 'accessorySunglasses', 'lowerBodyTrousers', 'upperBodyTshirt', 
        'upperBodyOther', 'upperBodyVNeck', //
        'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 
        'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple',
        'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack', 'lowerBodyBlue', 
        'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPurple',
        'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue', 'hairBrown', 'hairGreen', 
        'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 'footwearBlack', 
        'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple',
        'footwearRed', 'footwearWhite', 'footwearYellow', 'accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 
        'hairBald', 'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 
        'carryingFolder', 'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief', 'lowerBodyLongSkirt', 
        'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'upperBodyNoSleeve', 
        'hairShort', 'footwearStocking', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 
        'upperBodyThickStripes']
        '''
        from scipy.io import loadmat
        matname=os.path.join(root_path,"PETA.mat")
        data = loadmat(matname)
        bs=[] 
        dataset={}
        dataset['image'] = []
        dataset['att'] = []
        dataset['att_name'] = []
        #data['peta'][0][0][0]--1'th coloumn: index of image, 2'th coloumn: global person identity,
        #3'th coloumn: name index of each single dataset.  4'th coloumn: person identitity in the orignal datasets.
        #5-109'th coloumns: attribute annotations
            
        for idx in range(5):
            train = (data['peta'][0][0][3][idx][0][0][0][0][:,0]-1).tolist()
            val = (data['peta'][0][0][3][idx][0][0][0][1][:,0]-1).tolist()
            test = (data['peta'][0][0][3][idx][0][0][0][2][:,0]-1).tolist()
            trainval = train + val  
        
        for idx in range(105):
            dataset['att_name'].append(data['peta'][0][0][1][idx,0][0])
        

        for idx in range(19000):
            dataset['image'].append('%05d.png'%(idx+1))
            dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())
        
        #get_train
        dicts={}
        dicts["train"]=train
        dicts["val"]=val
        dicts["test"]=test
        #pdb.set_trace()
        for key, item in dicts.items():
            
            if phase=="test":
                if key=="test":
                    for i in item:
                        name=dataset['image'][i]
                        imagename=os.path.join(root_path,"images","images", dataset['image'][i])                 
                        index=np.nonzero(dataset['att'][i])
                        #pdb.set_trace()
                        caption=[]
                        for ite in index[0]:                       
                            at=dataset['att_name'][ite]
                            caption.append(at)
                        target, captions = self.get_one_target(caption) 
                        if save_root!=None:
                            f=open(os.path.join(save_root, name.split(".")[0]+".txt"), 'a') 
                            for tt in captions:
                                #pdb.set_trace()
                                f.write(tt)
                            f.write(str(dataset['att'][i])+";")
                            f.close() 
                            shutil.copy(imagename, os.path.join(save_root,name))               
                        #b0=self.path2rest(imagename, target, key,dataset['att'][i])
                        #bs.append(b0)
            else:
                for i in item:
                    name=dataset['image'][i]
                    imagename=os.path.join(root_path,"images","images", dataset['image'][i])                 
                    index=np.nonzero(dataset['att'][i])
                    caption=[]
                    for ite in index[0]:                       
                        at=dataset['att_name'][ite]
                        caption.append(at)
                    #print(caption)
                    target, captions = self.get_one_target(caption) 

                    if key !="test":
                        #pdb.set_trace()
                        if save_root!=None:
                            f=open(os.path.join(save_root, name.split(".")[0]+".txt"), 'a') 
                            for tt in captions:
                                f.write(tt)
                            f.write(str(dataset['att'][i])+";")
                            f.close() 
                            shutil.copy(imagename, os.path.join(save_root,name)) 
                    b0=self.path2rest(imagename, target, key,dataset['att'][i])
                    bs.append(b0)
                
        #pdb.set_trace()
        return bs

    def get_test(self, root_path,save_root=None,phase=None):
            '''
            dataset['att_name']
            ['personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBackpack', 
            'carryingOther', 'lowerBodyCasual', 'upperBodyCasual', 'lowerBodyFormal', 'upperBodyFormal', 
            'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans', 'footwearLeatherShoes', 'upperBodyLogo', 
            'hairLong', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 
            'carryingNothing', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 
            'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneaker', 
            'upperBodyThinStripes', 'accessorySunglasses', 'lowerBodyTrousers', 'upperBodyTshirt', 
            'upperBodyOther', 'upperBodyVNeck', //
            'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 
            'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple',
            'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack', 'lowerBodyBlue', 
            'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPurple',
            'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue', 'hairBrown', 'hairGreen', 
            'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 'footwearBlack', 
            'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple',
            'footwearRed', 'footwearWhite', 'footwearYellow', 'accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 
            'hairBald', 'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 
            'carryingFolder', 'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief', 'lowerBodyLongSkirt', 
            'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'upperBodyNoSleeve', 
            'hairShort', 'footwearStocking', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 
            'upperBodyThickStripes']
            '''
            from scipy.io import loadmat
            matname=os.path.join(root_path,"PETA.mat")
            data = loadmat(matname)
            bs=[] 
            dataset={}
            dataset['image'] = []
            dataset['att'] = []
            dataset['att_name'] = []
            #data['peta'][0][0][0]--1'th coloumn: index of image, 2'th coloumn: global person identity,
            #3'th coloumn: name index of each single dataset.  4'th coloumn: person identitity in the orignal datasets.
            #5-109'th coloumns: attribute annotations
                
            for idx in range(5):
                train = (data['peta'][0][0][3][idx][0][0][0][0][:,0]-1).tolist()
                val = (data['peta'][0][0][3][idx][0][0][0][1][:,0]-1).tolist()
                test = (data['peta'][0][0][3][idx][0][0][0][2][:,0]-1).tolist()
                trainval = train + val  
            
            for idx in range(105):
                dataset['att_name'].append(data['peta'][0][0][1][idx,0][0])
            

            for idx in range(19000):
                dataset['image'].append('%05d.png'%(idx+1))
                dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())
            
            #get_train
            dicts={}
            dicts["train"]=train
            dicts["val"]=val
            dicts["test"]=val
            #pdb.set_trace()
            for key, item in dicts.items():
                
                if phase=="test":
                    if key=="test":
                        for i in item:
                            imagename=os.path.join(root_path,"images","images", dataset['image'][i])                 
                            index=np.nonzero(dataset['att'][i])
                            pdb.set_trace()
                            caption=[]
                            for ite in index[0]:                       
                                at=dataset['att_name'][ite]
                                caption.append(at)
                            target, captions = self.get_one_target(caption)               
                            # b0=self.path2rest(imagename, target, key)
                            # bs.append(b0)
                            if save_root!=None:
                                f=open(os.path.join(save_root, name.split(".")[0]+".txt"), 'a') 
                                for tt in captions:
                                    f.write(tt+";"+str(dataset['att'][i]))
                                f.close() 
                                shutil.copy(imagename, os.path.join(save_root,name)) 
                else:
                    for i in item:
                        name=dataset['image'][i]
                        imagename=os.path.join(root_path,"images","images", dataset['image'][i])                 
                        index=np.nonzero(dataset['att'][i])
                        caption=[]
                        for ite in index[0]:                       
                            at=dataset['att_name'][ite]
                            caption.append(at)
                        #print(caption)
                        target, captions = self.get_one_target(caption) 

                        if key !="test":
                            #pdb.set_trace()
                            if save_root!=None:
                                f=open(os.path.join(save_root, name.split(".")[0]+".txt"), 'a') 
                                for tt in captions:
                                    f.write(tt)
                                f.close() 
                                shutil.copy(imagename, os.path.join(save_root,name)) 
                        # b0=self.path2rest(imagename, target, key)
                        # bs.append(b0)
                    
            #pdb.set_trace()
            return bs




if __name__ == '__main__':

    root_path="/raid2/yue/datasets/Attribute-Recognition/PETA/PETA_select/PETAdata/"
    #save_root="../../dataset/PETA_select_35/"
    save_root="../../dataset/PETAdata/PETA_train_label/"
    keys=["gender","upperbody_1","upperbody_2","upperbody_3","lowerbody_1","lowerbody_2","lowerbody_3","age","hair_1","hair_2","foot_1","foot_2", "carry","accessory"]
    
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
        os.makedirs(save_root)
    else:
        os.makedirs(save_root)
   
    #加载数据
    #pdb.set_trace()
    petadata=petabaseDataset(root_path)
    classes=petadata.classes
    templates=petadata.templates
    bs=petadata.read_mat(root_path, save_root,phase="train") #save_root,
    #bs=petadata.read_mat(root_path,save_root, phase="train") 
