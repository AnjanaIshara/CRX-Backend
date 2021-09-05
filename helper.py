import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

class_names = ['normal', 'viral', 'covid']
test_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(size=(224, 224)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def initialize():
    torch.manual_seed(0)
    print('Using PyTorch version', torch.__version__)
    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
    resnet18.load_state_dict(torch.load('covid_classifier.pt'))
    resnet18.eval()
    return resnet18

def predict_image_class(image_path,resnet18):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    # Please note that the transform is defined already in a previous code cell
    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name