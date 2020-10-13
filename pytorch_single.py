import torch
from torchvision import datasets, models, transforms
#import mb2
import cv2
import sys
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)

model=torch.load('/home/cjkim/pytorch_test2/model/savetest.pth',map_location=DEVICE)

model.eval()

input_image = Image.open(sys.argv[1])

preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor()
                        ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    result = model(input_batch)

pr = torch.argmax(torch.nn.functional.softmax(result[0], dim=0))
result1 = torch.nn.functional.softmax(result[0], dim=0)
round_result = round(float(result1[pr]),4)
print(f"conf : {round_result}, result : {pr}")

