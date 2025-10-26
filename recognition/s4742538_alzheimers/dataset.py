"""
Containing the data loader for loading and preprocessing your data
"""
from torchvision import transforms, datasets # type: ignore
from torch.utils.data import DataLoader, random_split, Subset, Dataset  # type: ignore
import matplotlib # type: ignore 
matplotlib.use("Agg") # to work in wsl
import matplotlib.pyplot as plt  # type: ignore
import numpy as np # type: ignore
from parameters import TRAIN_DIR, TEST_DIR, CHANNELS, IMAGE_SIZE, BATCH_SIZE, SAMPLE_IMAGE_FILENAME
import os
from sklearn.model_selection import train_test_split # type: ignore 
from collections import defaultdict 
from PIL import Image # type: ignore
import torch # type: ignore

# reduce image size, convert to tensor, and then normalise
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])  # pixel values [-1, 1]
])

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS),
    # Add aggressive augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), #translates and zoom up to 10%
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # random ajustments to brightness
    #transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)), # randomly crops a little bit
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05))  # erases a rectangle region
])

# Get Images
train_val_data = datasets.ImageFolder(root=TRAIN_DIR, transform=None) # do transform in class
train_class_to_idx = train_val_data.class_to_idx # explicitly assigns labels
print (f"labels assigned: {train_class_to_idx}")

train_val_person_to_slices = defaultdict(list) #{person1: [1,2,3], person2: [4,5,6]...}
train_val_person_labels = {}
for path, label in train_val_data.samples:
    pid = os.path.basename(path).split("_")[0]
    train_val_person_to_slices[pid].append(path)
    train_val_person_labels[pid] = label

for pid in train_val_person_to_slices: # make sure its in order
    train_val_person_to_slices[pid] = sorted(train_val_person_to_slices[pid])

# now for test set
test_data = datasets.ImageFolder(root=TEST_DIR, transform=None) # do transform in class

test_person_to_slices = defaultdict(list)
test_person_labels = {}

for path, label in test_data.samples:
    pid = os.path.basename(path).split("_")[0]
    test_person_to_slices[pid].append(path)
    test_person_labels[pid] = label

for pid in test_person_to_slices:
    test_person_to_slices[pid] = sorted(test_person_to_slices[pid])

# # extract the person_ids
# image_paths = [path for path, _ in train_val_data.samples]
# person_ids = [os.path.basename(path).split("_")[0] for path in image_paths]

# person_to_indices = defaultdict(list) 
# for idx, pid in enumerate(person_ids):
#     person_to_indices[pid].append(idx) # stores (persons_id, [list of index's for that persons images])

train_val_unique_person_ids = list(train_val_person_to_slices.keys())
test_person_ids = list(test_person_to_slices.keys())

train_person_ids, val_person_ids = train_test_split(
    train_val_unique_person_ids, 
    test_size=0.2, 
    random_state=42,
    stratify=[train_val_person_labels[pid] for pid in train_val_unique_person_ids] # makes it evenly split
)

print(f"unique persons {len(train_val_unique_person_ids)}")
print(f"train persons {len(train_person_ids)}")
print(f"val persons {len(val_person_ids)}")


class PersonDataset(Dataset):
    def __init__(self, person_ids, person_to_slices, person_labels, transform):
        self.person_ids = person_ids
        self.person_to_slices = person_to_slices
        self.person_labels = person_labels
        self.transform = transform

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        pid = self.person_ids[idx]
        slice_paths = self.person_to_slices[pid]
        label = self.person_labels[pid]

        slices = []
        for path in slice_paths:
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            img = img.squeeze(0)  # remove old greyscale channel dim
            slices.append(img)

        # Stack into (20, H, W)
        volume = torch.stack(slices, dim=0)
        return volume, label


train_dataset = PersonDataset(train_person_ids, train_val_person_to_slices, train_val_person_labels, train_transform)
val_dataset = PersonDataset(val_person_ids, train_val_person_to_slices, train_val_person_labels, transform)
test_dataset = PersonDataset(test_person_ids, test_person_to_slices, test_person_labels, transform)

# sanity check
print(f"train size {len(train_dataset)}")
print(f"val size {len(val_dataset)}")
print(f"test size {len(test_data)}")

# Make data loaders for each
train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True,
                          persistent_workers=True)
val_loader = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        persistent_workers=True)
test_loader = DataLoader(test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False,
                         num_workers=2,
                         pin_memory=True,
                         persistent_workers=True)


"""Visualise first image as a sanity check"""
# data_iter = iter(train_loader)
# images, labels = next(data_iter)

# # look at the first image
# img = images[0, 0]  # remove batch & channel dims
# label = labels[0].item()

# img = img * 0.5 + 0.5 # un-normalise

# # Convert to numpy and plot
# plt.imshow(img.squeeze(0).numpy(), cmap='gray')
# plt.savefig(SAMPLE_IMAGE_FILENAME)
# print(f"Label {label}")


print(f"Train labels: {sorted(train_val_data.class_to_idx.keys())}")
print(f"Test labels: {sorted(test_data.class_to_idx.keys())}")
print(f"Train unique labels: {np.unique([train_val_person_labels[p] for p in train_val_unique_person_ids])}")
print(f"Test unique labels: {np.unique([test_person_labels[p] for p in test_person_ids])}")


train_set = set(train_person_ids)
test_set = set(test_person_ids)
overlap = train_set & test_set
print(f"overlap: {overlap}") # no overlapping person ids