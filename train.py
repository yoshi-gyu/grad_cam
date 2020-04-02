# https://www.noconote.work/entry/2019/01/12/231723

# %matplotlib inline
import urllib
import pickle
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from grad_cam import GradCam

def train():
    labels = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_image = Image.open("images/guinea-pig-242520_640.jpg")
    # test_image = Image.open("images/cavy2.jpg")
    test_image_tensor = (transform((test_image))).unsqueeze(dim=0)

    image_size = test_image.size
    print("image size: ", image_size)

    # plt.imshow(test_image)

    model = models.vgg19(pretrained=True)

    grad_cam = GradCam(model)

    feature_image = grad_cam(test_image_tensor).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)

    pred_idx = model(test_image_tensor).max(1)[1]
    print("pred: ", labels[int(pred_idx)])
    # plt.title("Grad-CAM feature image")
    # plt.imshow(feature_image.resize(image_size))
    (feature_image.resize(image_size)).save("images/result_guinea-pig-242520_640.jpg")
    # (feature_image.resize(image_size)).save("images/result_cavy2.jpg")

if __name__ == '__main__':
    train()    