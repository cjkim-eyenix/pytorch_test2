from torchvision import datasets, models, transforms
#import mb2
import cv2
import numpy as np
import torch
import requests
import os
import shutil  
#shutil.copy 함수를 사용하기 위한 import

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('/home/cjkim/pytorch_test2/model/savetest.pth',map_location=DEVICE)

dir = "/home/cjkim/pytorch_test2/classification"

#expect = 1    
#평가 대상

model.eval()
#평가 모드로 전환

if torch.cuda.is_available():
    model.to('cuda')
    device = 'cuda'
else:
    device = 'cpu'

correct_count = 0

########### classification 디렉토리에 만들어둔 폴더들 때문에 에러가 나는듯 하여 추가함
########################################### file list에서 원하는 확장자 파일만 찾기
file_list = os.listdir(dir)
file_list_jpeg = [file for file in file_list if file.endswith(".JPEG")]
################################################################################


for image in file_list_jpeg:

    orig_image1 = cv2.imread(dir+"/"+image)
    
    to_pil = transforms.ToPILImage()
    #텐서 또는 ndarray를 PL 이미지로 변환
    orig_image = to_pil(orig_image1)
    
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()
                                ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #.Compose : 여러 변환들을 함께 컴파일
    #.Resize : 이미지 사이즈를 원하는 사이즈로 변환
    #.ToTensor : 이미지 데이터를 텐서로 변환
    #.Normalize : 채널 정규화

    jpeg_image = image
    #원본 .jpeg 이미지 변수에 담아두기

    image = trans(orig_image)

    image = image.unsqueeze(0)
    #차원을 줄일때 사용 : 기본 데이터는 공유함
    image = image.to(device)
            
    with torch.no_grad(): #Autograd(자동미분) 비활성화
        result = model(image) #model에 입력을 넣어 출력을 구함

    print("-"*50)
    pr = torch.argmax(torch.nn.functional.softmax(result[0], dim=0))
    result1 = torch.nn.functional.softmax(result[0], dim=0)
    #.argmax : 입력 텐서 내 모든 원소의 최댓값 지수를 반환
    #.nn.functional.softmax : 소프트맥스(확률분포) 수행

    round_result = round(float(result1[pr]),4)
    #round : 가까운 짝수로 반올림

    print(f"conf : {round_result}, result : {pr}")
    
#    if pr == expect :
#        correct_count +=1
    
#print(f"acc : {correct_count}/{len(os.listdir(dir))}")

    src = '/home/cjkim/pytorch_test2/classification/' + jpeg_image
    #src : shutil.copy할 소스
    #dst : shutil.copy될 도착형태

    if pr == 0 :
        dst = '/home/cjkim/pytorch_test2/classification/balloon/' + jpeg_image
        shutil.copy(src, dst)
    elif pr == 1:
        dst = '/home/cjkim/pytorch_test2/classification/banana/' + jpeg_image
        shutil.copy(src, dst)        
    elif pr == 2:
        dst = '/home/cjkim/pytorch_test2/classification/bell/' + jpeg_image
        shutil.copy(src, dst)        
    elif pr == 3:
        dst = '/home/cjkim/pytorch_test2/classification/cdplayer/' + jpeg_image
        shutil.copy(src, dst)        
    elif pr == 4:
        dst = '/home/cjkim/pytorch_test2/classification/cleaver/' + jpeg_image
        shutil.copy(src, dst)  
    elif pr == 5:
        dst = '/home/cjkim/pytorch_test2/classification/cradle/' + jpeg_image
        shutil.copy(src, dst)
    elif pr == 6:
        dst = '/home/cjkim/pytorch_test2/classification/crane/' + jpeg_image
        shutil.copy(src, dst) 
    elif pr == 7:
        dst = '/home/cjkim/pytorch_test2/classification/daisy/' + jpeg_image
        shutil.copy(src, dst)
    elif pr == 8:
        dst = '/home/cjkim/pytorch_test2/classification/helmet/' + jpeg_image
        shutil.copy(src, dst)
    elif pr == 9:
        dst = '/home/cjkim/pytorch_test2/classification/speaker/' + jpeg_image
        shutil.copy(src, dst)
