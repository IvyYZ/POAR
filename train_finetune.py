import torch
import clip, clipS
import copy
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
import pdb

def main(hparams):
    clp, preprocess = clip.load("ViT-B/16", device='cpu')
    
    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size
    
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
    
    # # Visual pre-trained weights
    visual_state_dict = clp.visual.state_dict()
    visual_state_dict.pop('class_embedding')
    model.model.visual.load_state_dict(visual_state_dict, strict=False)
    
    model_path="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-14test/lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt"
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    #pdb.set_trace()
    dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=100)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
