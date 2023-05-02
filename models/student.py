from re import T
from tkinter import N
from tkinter.tix import TList
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import yaml
import copy
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from .model import CLIP
import pdb
import clip
from clipS.model import VisionTransformer
from PIL import Image
from .MultiSimilarityLoss import MultiSimilarityLoss

class CLIPWrapper(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 config: dict,
                 minibatch_size: int
                 ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model_name = model_name
        self.model = CLIP(**config)
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name

        self.automatic_optimization = False
    
    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)
        #pdb.set_trace()
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

        return (dataset_size // num_devices) * self.trainer.max_epochs

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_text(mb), dim=1)
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-2,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=500
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CustomCLIPWrapper(CLIPWrapper):
    def __init__(self,
                 #image_encoder,
                 text_encoder,
                 minibatch_size,
                 learning_rate=1e-4,
                 kl_coeff=1.0,
                 avg_word_embs=False
                 ):
        # with open('models/configs/RN.yaml') as fin:
        #     config = yaml.safe_load(fin)['RN50']
        # super().__init__('RN50', config, minibatch_size)

        with open('models/configs/ViT.yaml') as fin:
            config = yaml.safe_load(fin)['ViT-B/16']
        super().__init__('ViT-B/16', config, minibatch_size)

        self.model.visual = VisionTransformer(input_resolution=224, patch_size=16, width=768, layers=6, heads=12, output_dim=512)
        self.model.transformer = text_encoder
        self.learning_rate = learning_rate
        self.avg_word_embs = avg_word_embs
        self.sink_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.minibatch_size = minibatch_size
        
        self.kl_coeff = kl_coeff 
        self.txtpath="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-6TScls/data/cls_35.txt"
        #self.txtpath="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-6TScls/data/cls_pa100k.txt" #cls_35.txt
    def get_cls_35(self,k):

        k_value={0:"hair",1:"age",2:"gender",3:"carry",4:"accessory", 5:"foot", 6:"upperbody", 7:"lowerbody"} #PETA
        #k_value={0:"gender",1:"age",2:"body",3:"accessory",4:"carry", 5:"upperbody", 6:"lowerbody", 7:"foot"} #pa100k
        
        fr=open(self.txtpath,"r")
        dic=eval(fr.read())
        cur_cls=dic[k_value[k]]
        return cur_cls

    def get_text_label(self, indices, text1, text_split2, k):
                     
        text_nolap, dictI_gt=self.get_no_overlap_text(text_split2)
        target_classes,weight=self.get_target_classes(indices, text_nolap, dictI_gt)
        try:
            ground_truth_T,text_nolap_g=self.get_global_sametext(text1) 
        except:
            pdb.set_trace()
            ground_truth_T,text_nolap_g=self.get_global_sametext(text1)
        
        n=1
        text_mbs = text_nolap_g.cuda()
        text_nolap_mbs = text_nolap.cuda()

        return text_mbs,text_nolap_mbs,indices,ground_truth_T,target_classes,weight
    
    
    def tokenize(self, text):
        tokenized_text = torch.stack([clip.tokenize(t, truncate=True) for t in text])
        tokenized_text_split=[]
        for t in text:
            tmp = []
            for item in t:            
                description2=list(item.split("."))
                description2=description2[:len(description2)-1]
                token_t= clip.tokenize(description2, truncate=True)
                tmp.append(token_t)
            tokenized_text_split.append(tmp)
        return tokenized_text, tokenized_text_split
    

    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()   
        image, text, keys = train_batch
        
        for i in range(len(image)):
            debug_image = image[i].permute(1, 2, 0)
            debug_image = debug_image - debug_image.min()
            debug_image = debug_image / debug_image.max() * 255.
            
            im = Image.fromarray(np.uint8(debug_image.cpu().numpy()), mode='RGB')
            im.save(f'debug/debug_{i:02d}.jpg')
        
        tokenized_text, tokenized_text_split = self.tokenize(text)
        
        image = image.cuda()

        optimizer.zero_grad()

        # image loss
        image_features, attention = self.model.encode_image(image, k_num=8)
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
            #pdb.set_trace()
            indices, text1, text_split2 = self.filter_nontext(tokenized_text,tokenized_text_split, k)
            if len(text1)==0:
                continue
            text_mbs,text_nolap_mbs,indices,ground_truth_T,target_classes,weight=self.get_text_label(indices, text1, text_split2, k)
            txt = self.encode_text(text_mbs)
            txt = txt / txt.norm(dim=-1, keepdim=True)

            txt_nolap = self.encode_text(text_nolap_mbs)
            txt_nolap = txt_nolap / txt_nolap.norm(dim=-1, keepdim=True)


            target_classes=target_classes.type_as(image_features).long().cuda()
            src_logits=image_features[indices][:, k] @ txt_nolap.t()
            src_logits = src_logits * self.model.logit_scale.exp()
            
            loss_ik = self.masked_out_cross_entropy(src_logits, target_classes,weight)
            loss_tk = self.masked_out_cross_entropy(src_logits.t(), target_classes.t())
            
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
        self.log_dict({'part': k,'loss_i': loss_i.detach() / 8,'loss_t': loss_t.detach() / 8, 'acc_i': acc_i ,'acc_t': acc_t}, prog_bar=True)

        loss = (loss_i + loss_t) / 16 #+ loss_g/8
        self.manual_backward(loss)  
        
        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))
        torch.cuda.empty_cache()
 

    def encode_text(self, inputs, teacher=False):

        inference = self.teacher if teacher else self.model
        x = inference.token_embedding(inputs).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + inference.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        #x = x if teacher else x.type(torch.float16)
        x = inference.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = inference.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x if teacher else x.type(torch.float16)
        pdb.set_trace()
        x = x[torch.arange(x.shape[0]), inputs.argmax(dim=-1)] @ self.model.text_projection
        x=x.type(self.dtype)

        return x

    def compute_similarities(self, I_emb, T_emb):
        sim_ii, sim_tt = I_emb @ I_emb.t(), T_emb @ T_emb.t()
        sim_it, sim_ti = I_emb @ T_emb.t(), T_emb @ I_emb.t()
        return sim_ii, sim_tt, sim_it, sim_ti

    def update_teacher(self):
        i=0
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            i+=1
            #print("i",i)
            #print("teacher:",teacher.shape)
            #print("student:",student.shape)
            if(teacher.shape==student.shape):
                teacher.data.copy_(self.ema(teacher.data, student.data))

    def ema(self, s, t):
        return s * (1 - 0.999) + t * 0.999

    def forward(self, images, text):
        pdb.set_trace()
        logits = F.normalize(self.model.encode_image(images), dim=1) @ F.normalize(self.encode_text(text), dim=1).t() * self.model.logit_scale.exp()
        return logits, logits.t()

    # Sourced from: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def configure_optimizers(self):
        lr = self.learning_rate

        # * frozen CLIP model
        update_modules, update_params = [], []
        frozen_modules, frozen_params = [], []
        for n, p in self.named_parameters():
            if 'model.visual' in n:
                update_modules.append(n)
                update_params.append(p)
            else:
                frozen_modules.append(n)
                frozen_params.append(p)
                p.requires_grad = False

        # optimizer = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr=lr,
        #     momentum=0.9
        # )
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, weight_decay=1e-4)
        
        print(self.num_training_steps)
        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        #pdb.set_trace()
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=100
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    #2022.6.14 xiaodui 相同text文本赋予同样的label
    def filter_nontext(self, text, text_split, k=0):
        la=[] 
        text_split2=[]
        #过滤空描述的图像文本对 
        indices = []
        text1=[]
        for i in range(self.minibatch_size):
            if torch.count_nonzero(text[i][k])>3:          
                indices.append(i)
                text1.append(text[i][k].unsqueeze(0))
                text_split2.append(text_split[i][k])
        if len(text1)==0:
            return indices,text1,text_split2 
        else:
            return torch.tensor(indices), torch.cat(text1, dim=0), text_split2
      
      

    def get_nolap_label_txt(self,txt,T_label,ground_truth_T):
        label_loc={}
        for i in range(len(T_label)):
            label_loc[T_label[i]]=i
            if i==0:
                txt_nolap=txt[0][T_label[i]].unsqueeze(0)              
            else:
                txt_nolap=torch.cat((txt_nolap,txt[0][T_label[i]].unsqueeze(0) ),0)  

        target_classes=torch.zeros(len(txt[0]),len(T_label))

        bb=ground_truth_T.cpu()
        for i in range(len(ground_truth_T)):
            p1=label_loc[int(bb[i])]
            target_classes[i][p1]=1

        return txt_nolap, target_classes, label_loc

    #2022.7.5 xiaodui 相同text文本赋予同样的label
    def get_no_overlap_text_0(self,texts):
        dictI_gt={}
        text_des=[] 
        testsL=[] 
        #pdb.set_trace()    
        for i in range(len(texts)):           
            tx0=texts[i].cpu().numpy()  
            #index=np.argwhere(tx0==269)  #269是句号的code, 查找句号的索引
            tx=tx0.tolist()
            for t in range(len(index)):
                if t==0: 
                    tes=tx[1:int(index[t])+1]  #去除开始code
                    textt=texts[i][1:int(index[t])+1]
                    if i==0:
                        text_nolap=textt.unsqueeze(0)
                        text_des.append(tes)
                                        
                else:
                    tes=tx[int(index[t-1]+1):int(index[t])+1] #去除ending code
                    textt=texts[i][int(index[t-1]+1):int(index[t])+1]
                if tes not in text_des:                      
                    text_des.append(tes)
                    #pdb.set_trace()
                    text_nolap=torch.cat((text_nolap,textt.unsqueeze(0)),0)

        for i in range(len(texts)):     #遍历原始的text，对每个text的每句话在text_des中找对应的位置。
            tx0=texts[i].cpu().numpy()  
            index=np.argwhere(tx0==269)  
            tx=tx0.tolist()
            test1L=[]          
            for t in range(len(index)):
                if t==0:
                    tes=tx[1:int(index[t])+1]  #去除开始code 
                else:
                    tes=tx[int(index[t-1]+1):int(index[t])+1] #去除ending code                
                key_I="I_"+str(i)
                value_T="T_"+str(text_des.index(tes))+","
                if key_I not in dictI_gt.keys():
                    dictI_gt[key_I]=value_T
                else:
                    dictI_gt[key_I]+=value_T
                
                test1L.append(tes)
            testsL.append(test1L)

        return text_nolap,dictI_gt


    #2022.7.5 xiaodui 相同text文本赋予同样的label
    def get_no_overlap_text1(self,texts):
        dictI_gt={}
        text_des=[] 
        testsL=[] 
        #pdb.set_trace()
        for i in range(len(texts)):           
            tx0=texts[i].cpu().numpy()  
            tx=tx0.tolist()
            textt=texts[i]

            for j in range(len(tx)):           
                if i==0 and j==0:
                    text_nolap=textt[j].unsqueeze(0)
                    text_des.append(tx[j])
                                        
                if tx[j] not in text_des:                      
                    text_des.append(tx[j])
                    text_nolap=torch.cat((text_nolap,textt[j].unsqueeze(0)),0)

        for i in range(len(texts)):     #遍历原始的text，对每个text的每句话在text_des中找对应的位置。
            textt=texts[i]
            tx=texts[i].cpu().numpy().tolist()  
            test1L=[]          
            for item in tx:               
                key_I="I_"+str(i)
                value_T="T_"+str(text_des.index(item))+","
                if key_I not in dictI_gt.keys():
                    dictI_gt[key_I]=value_T
                else:
                    dictI_gt[key_I]+=value_T
                
        return text_nolap,dictI_gt

    #2022.7.5 xiaodui 相同text文本赋予同样的label
    def get_no_overlap_text(self,texts):
        dictI_gt={}
        text_des=[] 
        testsL=[] 
        #pdb.set_trace()
        for i in range(len(texts)):           
            tx0=texts[i].cpu().numpy()  
            tx=tx0.tolist()
            textt=texts[i]

            for j in range(len(tx)):           
                if i==0 and j==0:
                    text_nolap=textt[j].unsqueeze(0)
                    text_des.append(tx[j])
                                        
                if tx[j] not in text_des:                      
                    text_des.append(tx[j])
                    text_nolap=torch.cat((text_nolap,textt[j].unsqueeze(0)),0)

        for i in range(len(texts)):     #遍历原始的text，对每个text的每句话在text_des中找对应的位置。
            textt=texts[i]
            tx=texts[i].cpu().numpy().tolist()  
            test1L=[]          
            for item in tx:               
                key_I="I_"+str(i)
                value_T="T_"+str(text_des.index(item))+","
                if key_I not in dictI_gt.keys():
                    dictI_gt[key_I]=value_T
                else:
                    dictI_gt[key_I]+=value_T
                
        return text_nolap,dictI_gt

   
    def get_target_classes(self, indices, text_nolap, dictI_gt):
        target_classes=torch.zeros(len(indices),len(text_nolap))
        count_class={}
        weight=torch.ones(len(text_nolap))
        weight=weight.cuda()
        for i in range(len(text_nolap)):
            count_class[i]=0

        for key, item in dictI_gt.items():
            t1=int(key.split("_")[1])
            pp=item.split(",")
            for ite in pp:
                if ite !='':
                    t2=int(ite.split("_")[1])
                    try:
                        target_classes[t1][t2]=1
                        count_class[t2]+=1
                    except:
                        pdb.set_trace()
        #根据个数给对应标签加权重
        #count_class=Counter(list(dictI_gt.values()))
        maxN=max(list(count_class.values()))
        for i in range(len(text_nolap)):
            value=count_class[i]              
            weight[i]=1/value   
        return target_classes,weight
    
    def get_global_sametext(self,text1):
        dictT_gt={}
        Tl=[]
        la=[]
        text2=text1[0].unsqueeze(0)
        #相同text文本赋予同样的label， dictT_gt 不同类别的好几句话作为一个整体，global训练        
        for i in range(max(int(len(text1)/2),1)):                
            if "I_"+str(i) not in dictT_gt.keys(): #从第1个位置开始筛选，一直反复
                dictT_gt['I_'+str(i)]='T_'+str(i) 
                
                Tl.append(i)                     #Tl是不重复的标签            
                if i>0:
                    text2=torch.cat((text2,text1[i].unsqueeze(0)),0)

            for j in range(len(text1)):
                if i+j+1<len(text1):
                    key_I="I_"+str(i+j+1)

                    if text1[i].equal(text1[i+j+1]):
                        if key_I not in dictT_gt.keys():
                            value_T="T_"+str(i)
                            dictT_gt[key_I]=value_T
                            

                    if i==int(len(text1)/2)-1:
                        if key_I not in dictT_gt.keys():
                            value_T="T_"+str(i+j+1)
                            dictT_gt[key_I]=value_T                        
                            Tl.append(i+j+1)
                            text2=torch.cat((text2,text1[i+j+1].unsqueeze(0)),0)
        GTs=torch.tensor([int(dictT_gt["I_"+str(i)].split('_')[1]) for i in range(len(dictT_gt.items()))])
        label_loc={}
        for i in range(len(Tl)):
            label_loc[Tl[i]]=i
        ground_truth_T=torch.tensor([label_loc[int(GTs[i])] for i in range(len(GTs))]).cuda()
        return ground_truth_T,text2


    def masked_out_cross_entropy(self, src_logits, target_classes,weight_l_class=None):
        
        loss = 0
        #pdb.set_trace()
        num_pos = target_classes.sum(dim=-1) #行求和
        # If there is only one active positive label, then this will be ordinary cross entropy        
        #indices = torch.nonzero(num_pos < 2, as_tuple=True)[0]
        indices = torch.nonzero(num_pos==1, as_tuple=True)[0]
        try:
            if len(indices)>0:
                targets_one_pos = torch.argmax(target_classes[indices], dim=-1)
                #loss += F.cross_entropy(src_logits[indices], targets_one_pos,weight_l_class, reduction="sum")
                loss += F.cross_entropy(src_logits[indices], targets_one_pos,weight_l_class, reduction="sum")

            # If there are multiple positive labels, then we compute them one by one. Each time,
            # the other positive labels are masked out.
            indices = torch.nonzero(num_pos > 1, as_tuple=True)[0]
            for i in indices:
                t = target_classes[i]
                cnt = sum(t)
                loss_t = 0
                for j in torch.nonzero(t):
                    mask = (t == 0)
                    mask[j] = True
                    tgt = t[mask].argmax(dim=-1, keepdim=True)
                    #pdb.set_trace()
                    #loss_t += F.cross_entropy(src_logits[i:i+1, mask], tgt,weight_l_class, reduction="sum")
                    if weight_l_class!=None:
                        #pdb.set_trace()
                        loss_t += F.cross_entropy(src_logits[i:i+1, mask], tgt,weight_l_class[mask], reduction="sum")
                    else:
                        loss_t += F.cross_entropy(src_logits[i:i+1, mask], tgt, reduction="sum")
                loss += (loss_t / cnt)
            #pdb.set_trace()
            #loss = loss / len(src_logits)
        except:
            pdb.set_trace()
        return loss

    def build_mask(self):
        
        mask = torch.empty(self.minibatch_size, 204) #196+8
        mask.fill_(float("-inf"))
        mask[:,:8].fill_(1)  # zero out the lower diagonal
        return mask
    
