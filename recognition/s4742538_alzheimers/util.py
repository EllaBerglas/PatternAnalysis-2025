import torchvision.transforms as transforms
import os
from PIL import Image
from numpy import mean
import torch

transform = transforms.Compose([
    transforms.ToTensor()
])

# adapted from 
# https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/

def get_m_sd_images():
    means = []
    sds = []
    for filename in os.listdir("./data/AD_NC/train/AD/"):
        file_path = os.path.join("./data/AD_NC/train/AD/", filename)
        img = Image.open(file_path)  
        img_tr = transform(img)
        m, std = img_tr.mean([1,2]), img_tr.std([1,2])
        means.append(m)
        sds.append(std)

    means = torch.stack(means)
    sds = torch.stack(sds)
    overall_mean = means.mean(dim=0)
    overall_std = sds.mean(dim=0)

    # print mean and std
    print("mean and std before normalize:")
    print("Mean of the image:", overall_mean)
    print("Std of the image:", overall_std)
    """
    Results:
    Mean of the image: tensor([0.1114])
    Std of the image: tensor([0.2184])
    """

get_m_sd_images()