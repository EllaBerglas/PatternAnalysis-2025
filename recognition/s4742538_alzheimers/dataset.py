"""
Containing the data loader for loading and preprocessing your data
"""
import torch  # type: ignore
import os 
from torchvision import transforms, datasets # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore
import matplotlib # type: ignore 
matplotlib.use("Agg") # to work in wsl
import matplotlib.pyplot as plt  # type: ignore
import numpy as np # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

DATA_ROOT = "./data/AD_NC/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
# TRAIN_AD = os.path.join(TRAIN_DIR, "AD")
# TRAIN_NC = os.path.join(TRAIN_DIR, "NC")

TEST_DIR = os.path.join(DATA_ROOT, "test")
# use these for specific evaluations??
# TEST_AD = os.path.join(TEST_DIR, "AD")
# TEST_NC = os.path.join(TEST_DIR, "NC")

CHANNELS = 1
IMAGE_SIZE = 224
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])  # pixel values [-1, 1]
])

# Get Images
train_val_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_data = datasets.ImageFolder(root=TEST_DIR, transform=transform)

# extract validation set fro the train set
val_size = int(len(train_val_data) * 0.2)
train_size = len(train_val_data) - val_size
val_dataset, train_dataset = random_split(train_val_data, [val_size, train_size])

# sanity check
print(f"train size {len(train_dataset)}")
print(f"val size {len(val_dataset)}")
print(f"test size {len(test_data)}")

# Make data loaders for each
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

"""Visulise first image as a sanity check"""
data_iter = iter(train_loader)
images, labels = next(data_iter)

# look at the first image
img = images[0].squeeze(0)  # remove batch & channel dims
label = labels[0].item()

img = img * 0.5 + 0.5 # un-normalise

# Convert to numpy and plot
plt.imshow(img.numpy(), cmap='gray')
plt.savefig("sample_image.png")
print(f"Label {label}")