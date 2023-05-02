import torch
from tqdm import tqdm
import clip
import pdb
import numpy as np
from easydict import EasyDict
from sentence_transformers import SentenceTransformer
import os

os.environ['TORCH-HOME']='/raid2/yue/torch-model'

def get_pedestrian_metrics(gt_label,preds_probs, threshold=0.7):# threshold=0.45):
    #pdb.set_trace() 
    # print("-------------------------------------------------") 
    # print(preds_probs) 
    if threshold is None:
        pred_label = preds_probs
    else:
        pred_label = preds_probs > threshold
    # print(gt_label)
    # print(pred_label)
    # print("-------------------------------------------------")

    eps = 1e-20
    result = EasyDict()


    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.add_acc = (true_pos+true_neg) / (true_pos + false_pos + false_neg +true_neg+ eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)
    result.tp=true_pos
    result.tn=true_neg
    result.fn=false_neg
    result.fp=false_pos

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result, pred_label

def get_pedestrian_metrics0(gt_label, preds_probs,threshold=0.7):# threshold=0.45):
    #pdb.set_trace() 
    # print("-------------------------------------------------") 
    # print(preds_probs) 
    #pred_label = preds_probs > threshold
    pred_label = preds_probs
    # print(gt_label)
    # print(pred_label)
    # print("-------------------------------------------------")

    eps = 1e-20
    result = EasyDict()


    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.add_acc = (true_pos+true_neg) / (true_pos + false_pos + false_neg +true_neg+ eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps) #TP/TP+FP   pre
    instance_prec = intersect_pos / (true_pos + eps) 
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

    
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            #pdb.set_trace()
            #print(texts)
            #texts=texts[0].split(".")[0]
            texts = clip.tokenize(texts).cuda()  # tokenize
            #self.token_embedding(text.type(self.dtype))
            
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def text_classfier_weights(keys,classes,templates,model):
    zeroshot_weights={}
    for item in keys:
        #pdb.set_trace()
        k1 = item.split("_")[0]
        kn= item.split("_")
        
        if len(kn) > 1:
            k2 = item.split("_")[1]
            if k2 == '1':
                zeroshot_weights[item] = zeroshot_classifier(classes[k1], templates[k1],model)
            elif k2=='2':
                zeroshot_weights[item] = zeroshot_classifier(classes["color"], templates[k1],model)
            else:
                zeroshot_weights[item] = zeroshot_classifier(classes["style"], templates[k1],model)
        else:
            try:
                zeroshot_weights[item] = zeroshot_classifier(classes[k1], templates[k1],model)
            except:
                pdb.set_trace()
        
    return zeroshot_weights


def text_classfier_weights_all(keys,classes,templates,model):
    all_template=[]
    for item in keys:
        
        k1 = item.split("_")[0]
        kn= item.split("_")
        
        if len(kn) > 1:
            k2 = item.split("_")[1]
            if k2 == '1':
                curT=templates[k1]

            else:
                curT=templates[k1]
            for cont in curT:
                all_template.append(cont)
    #pdb.set_trace()
    zeroshot_weights = zeroshot_classifier(classes, all_template,model)
        
    return zeroshot_weights

    
def zeroshot_classifier_vtb(classnames, templates, model):
    with torch.no_grad():
        #pdb.set_trace()
        model2 = SentenceTransformer('all-mpnet-base-v2')
        zeroshot_weights = []
        texts = model2.encode(classnames)
        for text in tqdm(texts):
            #texts = clip.tokenize(classname).cuda()  # tokenize
            #pdb.set_trace()  
            text=(torch.tensor(text)).cuda()                    
            class_embeddings = model.word_embed(text)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #class_embedding = class_embeddings.mean(dim=0)
            #class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def text_classfier_weights_vtb(keys,classes,templates,model):
    zeroshot_weights={}
    texts={}
    for item in keys:
        #pdb.set_trace()
        k1 = item.split("_")[0]
        kn= item.split("_")
        
        if len(kn) > 1:
            k2 = item.split("_")[1]
            if k2 == '1':
                zeroshot_weights[item] =zeroshot_classifier_vtb(classes[k1], templates[k1],model)
            elif k2=='2':
                zeroshot_weights[item] = zeroshot_classifier_vtb(classes["color"], templates[k1],model)
            else:
                zeroshot_weights[item] = zeroshot_classifier_vtb(classes["style"], templates[k1],model)
        else:
            zeroshot_weights[item]= zeroshot_classifier_vtb(classes[k1], templates[k1],model)
        
    return zeroshot_weights

