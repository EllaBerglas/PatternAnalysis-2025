"""
Containing the data loader for loading and preprocessing your data
"""
import torch  # type: ignore
import os 
from torchvision import transforms, datasets # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

DATA_ROOT = "./data/AD_NC/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_AD = os.path.join(TRAIN_DIR, "AD")
TRAIN_NC = os.path.join(TRAIN_DIR, "NC")

TEST_DIR = os.path.join(DATA_ROOT, "test")
TEST_AD = os.path.join(TRAIN_DIR, "AD")
TEST_NC = os.path.join(TRAIN_DIR, "NC")

CHANNELS = 1
IMAGE_SIZE = 1024

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS), 
    transforms.ToTensor(), 
])

train_val_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_data = datasets.ImageFolder(root=TEST_DIR, transform=transform)

val_size = int(len(train_val_data) * 0.2)
train_size = len(train_val_data) - val_size

val_dataset, train_dataset = random_split(train_val_data, [val_size, train_size])

print(f"train size {len(train_dataset)}")
print(f"val size {len(val_dataset)}")
print(f"test size {len(test_data)}")