import torch
import torch.nn as nn
from torchvision import models
import clip
import math
import random

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device="cpu")
        
    def forward(self, x):
        with torch.no_grad():
            _, logits_per_image = self.clip_model(x)
        return logits_per_image

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, hparams, self).__init__()
        #self.resnet_model = models.resnet18(pretrained=True)
        #self.resnet_model.fc = nn.Linear(512, 1000)

        self.model_path="./lightning_logs/version_1_peta_b/checkpoints/epoch=99-step=33899.ckpt"  #train on PETA   
        # model_path="/raid2/yue/ReID/vision_language/train-CLIP-2th/train-CLIP-FT-14TScls_RAPv1/lightning_logs/version_6_best/epoch=83-step=21755.ckpt" # train on RAPv1  
        
        #load model
        self.hparams=hparams
        self.model=self.load_model(self.hparams,self.model_path)

    
    def load_model(self):
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

    def forward(self, x):
        x = self.resnet_model(x)
        return x

def kl_div_loss(outputs, targets):
    log_softmax_outputs = nn.functional.log_softmax(outputs, dim=1)
    softmax_targets = nn.functional.softmax(targets, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(log_softmax_outputs, softmax_targets)

def train_student_model(student_model, teacher_model, train_loader, optimizer, epoch):
    student_model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        # Get teacher predictions
        teacher_outputs = teacher_model(data).detach()

        # Get student predictions
        student_outputs = student_model(data)

        # Compute KL divergence loss
        loss = kl_div_loss(student_outputs, teacher_outputs)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Set up data loader
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), 
					   (0.2023, 0.1994, 0.2010))])),
batch_size=128, shuffle=True, num_workers=2)

#Set up models and optimizer
teacher_model = TeacherModel().cuda()
student_model = StudentModel().cuda()
optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


#Train student model for 10 epochs
for epoch in range(1, 11):
train_student_model(student_model, teacher_model, train_loader, optimizer, epoch)


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