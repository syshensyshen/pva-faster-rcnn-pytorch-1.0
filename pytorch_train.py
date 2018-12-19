# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 16:39
# @Author  : David Xu
# @modified by syshen @date: 2018/12/19 11:14
# @File    : train_back_model.py


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from random import shuffle
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Densenet_(nn.Module):
    def __init__(self, cls_num):
        super(Densenet_, self).__init__()
        Backstone = models.densenet201(pretrained=True)
        num_ftrs = Backstone.features
        self.BackstoneDim = 1920
        self.featureDim = 1280
        add_block = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.BackstoneDim, self.featureDim )),
            ('fc_relu', nn.LeakyReLU(inplace=True)),
            ('classifier', nn.Linear(self.featureDim, cls_num, bias=True))]))
        self.Backstone = Backstone.features
        self.add_block = add_block
        init.xavier_uniform(self.add_block.fc1.weight)
        init.constant(self.add_block.fc1.bias, 1.0)
        init.normal(self.add_block.classifier.weight, mean=0, std=0.001)
        init.constant(self.add_block.fc1.bias, 1.0)
        #init.xavier_uniform(self.add_block.classifier.weight)
        #init.constant(self.add_block.fc1.bias, 1.0)
        #init.normal(self.add_block.fc1.weight, mean=0, std=0.01)
        #init.constant(self.add_block.fc1.bias, 0.0)
        #init.normal(self.add_block.classifier.weight, mean=0, std=0.001)
        #init.constant(self.add_block.fc1.bias, 0.0)
        #init.kaiming_uniform_(self.add_block.fc1.weight)
        #init.constant(self.add_block.fc1.bias, 0.2)
        #init.kaiming_uniform_(self.add_block.classifier.weight)
        #init.constant(self.add_block.classifier.bias, 1.0)

    def forward(self, input):
        x = self.Backstone(input)
        H = x.shape[2]
        W = x.shape[3]
        x_relu = F.relu(x, inplace=True)
        x_relu = F.avg_pool2d(x_relu, (H, W), stride=1).view(x.size(0), -1)
        x = self.add_block(x_relu)
        return x

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/data/phase_1/images/back-train'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=48,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #inputs = inputs * 255
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()# gradient will acc if not make zeros

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if epoch > 45:
                        criterion = FocalLoss(num)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def validation(model):
    since = time.time()
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    res = np.zeros((len(class_names), len(class_names)),dtype=int)
    print('validation---------')
    for inputs, labels in dataloaders['val']:
        #inputs = inputs * 255
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(labels)):
            real = labels[i].item()
            pred = preds[i].item()
            res[real, pred] = res[real, pred] + 1
    print(class_names)
    print(res)
    print('training---------')
    res = np.zeros((len(class_names), len(class_names)),dtype=int)
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(labels)):
            real = labels[i].item()
            pred = preds[i].item()
            res[real, pred] = res[real, pred] + 1
    print(class_names)
    print(res)

num = 4

#model_ft = models.densenet121(pretrained=True)
model_ft = Densenet_(num)
#print(model_ft)
#input()

#for param in model_ft.parameters():
#    param.requires_grad = False
for param in model_ft.Backstone.parameters():
    param.requires_grad = False
for param in model_ft.add_block.parameters():
    param.requires_grad = True
#print(model_ft)

model_ft = model_ft.to(device)

#criterion = FocalLoss(num)
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
#optimizer_conv = optim.SGD(model_ft.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5)
ignored_params = list(map(id, model_ft.add_block.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model_ft.parameters())
optimizer_conv = optim.SGD([
                {'params': base_params, 'lr': 0.01},
                {'params': model_ft.add_block.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=4e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=40, gamma=0.1)
exp_lr_scheduler.step()
model_ft = train_model(model_ft, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=45)
validation(model_ft)
torch.save(model_ft, 'models/model-back-1.pt')

for param in model_ft.Backstone.parameters():
    param.requires_grad = True
for param in model_ft.add_block.parameters():
    param.requires_grad = True

model_ft = model_ft.to(device)

#criterion = FocalLoss(num)
#criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4)
optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.001},
                {'params': model_ft.add_block.parameters(), 'lr': 0.001},
                ], momentum=0.9, weight_decay=4e-4, nesterov=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler.step()
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=120)
validation(model_ft)
torch.save(model_ft, 'models/model-back-2.pt')
