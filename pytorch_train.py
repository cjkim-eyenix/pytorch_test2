from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
#import mb2

def train_model(model, criterion, optimizer, num_epochs, dset_sizes, dset_loaders):
    for epoch in range(num_epochs):
        print('-' * 35)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #.format : 앞에 {}에 값 넣기        

        model.train() 
        #학습모드로 전환

        running_loss = 0.0
        running_corrects = 0

        counter=0

        for data in dset_loaders:

            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.float().cuda()),Variable(labels.long().cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            #이전 epoch에서 계산되어 있는 파라미터의 gradient를 초기화
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
                
            loss = criterion(outputs, labels)
            #특정손실 함수에 따라 gradient 계산

            counter+=1
            
            loss.backward()     #역전파 (변화도 누적됨)
            optimizer.step()    ##매개변수 갱신(최적화)

            try:
                running_loss += loss.item()  #loss.item : loss의 스칼라값
                running_corrects += torch.sum(preds == labels.data)

            except:
                print('unexpected error, could not calculate loss or do a sum.')

        epoch_loss = running_loss / dset_sizes
        print('Loss: {:.4f}'.format(epoch_loss))
             
    return model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#epoch = 50
epoch = 10
batch_size = 32
num_class = 10

time_a = time.time()
print("-"*35)

print(f"Batch Size  :  {batch_size}")
print(f"Epoch       :  {epoch}")

model_ft = models.mobilenet_v2(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False                 #finetuning
                                                #윗부분 학습 안되도록 고정
                                                #마지막 fully-connected layer만 대체
model_ft.classifier = nn.Sequential(
                        nn.Linear(1280,100),
                        nn.ReLU(inplace=True),
                        nn.Linear(100,num_class))
#Sequential(<>) : <> 부분들 순서대로 값 전달 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#.RandomResizedCrop : 이미지 사이즈 변경
#.RandomHorizontalFlip : 이미지를 랜덤으로 수평 뒤집음

dsets = datasets.ImageFolder(("/home/cjkim/pytorch_test2/example"), data_transforms['train'])
#dsets 저장

dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=int(batch_size),shuffle=True)
#dsets-load, 미니배치 단위로 처리가능, 셔플가능

dset_sizes = len(dsets)
dset_classes = dsets.classes

criterion = nn.CrossEntropyLoss()
#다중분류 대표 손실함수

if torch.cuda.is_available():
    criterion.cuda()
    model_ft.cuda()
optimizer_ft = optim.Adam(model_ft.parameters())
#optim.Adam : 최소 손실 찾기위한 미분으로 구한 기울기에 따라 이동하는 이동 방식 중 하나

model_ft = train_model(model_ft, criterion, optimizer_ft,epoch,dset_sizes,dset_loaders)

torch.save(model_ft, "/home/cjkim/pytorch_test2/model/savetest.pth")
#모델 저장

print("TRAINING DONE")

