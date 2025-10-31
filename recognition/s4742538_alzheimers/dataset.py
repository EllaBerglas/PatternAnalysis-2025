"""
Containing the data loader for loading and preprocessing your data
This builds the datasets and data loaders for training, validation and test sets.
The training set includes 0.2% augmented data
The sets are split based on a person/ patient_id ensuring no data leakage
"""
import os
import sys
import numpy as np # type: ignore
from collections import defaultdict 
from PIL import Image # type: ignore
import torch # type: ignore
from torchvision import transforms, datasets # type: ignore
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore 

import matplotlib # type: ignore 
matplotlib.use("Agg") # to work in wsl (no ability to display)
import matplotlib.pyplot as plt  # type: ignore

from parameters import TRAIN_DIR, TEST_DIR, CHANNELS, IMAGE_SIZE, BATCH_SIZE, \
        NORMALISATION_M, NORMALISATION_SD, RANDOM_STATE, AUGMENTED_DS, AUG_PROB, CHECKS

# reduce image size, convert to tensor, and then normalise
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[NORMALISATION_M], std=[NORMALISATION_SD])  # pixel values [-1, 1]
])

# a transform with different data transformations is used for better generalisation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # change brightness and contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[NORMALISATION_M], std=[NORMALISATION_SD]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05))  # erases a small rectangal region
])

# Get Images
train_val_data = datasets.ImageFolder(root=TRAIN_DIR, transform=None)
test_data = datasets.ImageFolder(root=TEST_DIR, transform=None)

# print assigned labels/ classification
train_class_to_idx = train_val_data.class_to_idx
print (f"labels assigned: {train_class_to_idx}")

# build a dictionary of patient_ids and their segments
# should be 20 for each patient
train_val_person_to_slices = defaultdict(list) #{person1: [1,2,3], person2: [4,5,6]...}
train_val_person_labels = {}
for path, label in train_val_data.samples:
    pid = os.path.basename(path).split("_")[0]
    train_val_person_to_slices[pid].append(path)
    train_val_person_labels[pid] = label

for pid in train_val_person_to_slices: # ensure segments are in order
    train_val_person_to_slices[pid] = sorted(train_val_person_to_slices[pid])

# Same as above but for the test set
test_person_to_slices = defaultdict(list)
test_person_labels = {}

for path, label in test_data.samples:
    pid = os.path.basename(path).split("_")[0]
    test_person_to_slices[pid].append(path)
    test_person_labels[pid] = label

for pid in test_person_to_slices:
    test_person_to_slices[pid] = sorted(test_person_to_slices[pid])


# seperate the validation set from the test set stratified (proportionally) for person id's
# this ensures no data leakage
train_val_person_ids = list(train_val_person_to_slices.keys())
test_person_ids = list(test_person_to_slices.keys())

train_person_ids, val_person_ids = train_test_split(
    train_val_person_ids, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=[train_val_person_labels[pid] for pid in train_val_person_ids] # makes it proportional split
)

class PersonDataset(Dataset):
    def __init__(self, 
                 person_ids, 
                 person_to_slices, 
                 person_labels, 
                 transform, 
                 aug_transform=None, 
                 aug_prob=0.5):
        self.person_ids = person_ids
        self.person_to_slices = person_to_slices
        self.person_labels = person_labels
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        pid = self.person_ids[idx]
        slice_paths = self.person_to_slices[pid]
        label = self.person_labels[pid]

        # decide if this entry should be augmented (one patients 20 images)
        rand_num = np.random.rand()
        if (self.aug_transform is not None) and (rand_num < self.aug_prob):
            use_aug = True
        else:
            use_aug = False

        slices = []
        for path in slice_paths:
            img = Image.open(path).convert('L')
            if use_aug:
                img = self.aug_transform(img)
            else:
                img = self.transform(img)
            img = img.squeeze(0)  # remove old greyscale channel dim
            slices.append(img)

        volume = torch.stack(slices, dim=0) # Stack the slices into (20, H, W)
        return volume, label

# build each dataset
if AUGMENTED_DS: 
    train_dataset = PersonDataset(
        train_person_ids, 
        train_val_person_to_slices, 
        train_val_person_labels, 
        transform,
        aug_transform=train_transform,
        aug_prob=AUG_PROB
    )
else:
    train_dataset = PersonDataset(
        train_person_ids, 
        train_val_person_to_slices, 
        train_val_person_labels, 
        transform
    )

val_dataset = PersonDataset(
    val_person_ids, 
    train_val_person_to_slices, 
    train_val_person_labels, 
    transform
)

test_dataset = PersonDataset(
    test_person_ids, 
    test_person_to_slices, 
    test_person_labels, 
    transform
)

# sanity check, this should be the number of patients in each data set
print(f"train size {len(train_dataset)}")
print(f"val size {len(val_dataset)}")
print(f"test size {len(test_data)}")

# Make data loaders for each dataset
# Should be in the format (BATCH_SIZE, 20 (CHANNELS), IMAGE_HEIGHT, IMAGE_WIDTH)
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

def visualise_image(train_loader):
    """Visualise first image as a sanity check"""
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # look at the first image
    img = images[0, 5]  # 0th person, 5th image
    label = labels[0].item() # 0th persons label
    img = img * NORMALISATION_SD + NORMALISATION_M # un-normalise

    # Convert to numpy and plot
    plt.imshow(img.squeeze(0).numpy(), cmap='gray')
    plt.savefig("./sample_image.png")
    print(f"Label of figure {label}")

def check_normalisation(loader):
    """gets the largest and smallest values used after normalisation"""
    data_iter = iter(loader)
    images, labels = next(data_iter)

    print(f"  Min pixel value: {images.min().item():.4f}")
    print(f"  Max pixel value: {images.max().item():.4f}")

    sample = images[0, 0]  
    print(f"Sample slice min/max: {sample.min().item():.4f}, {sample.max().item():.4f}")

def check_classification_distirbution(train_person_ids, train_val_person_labels):
    """see distribution of each classification on the training and validation set"""
    ad_count = sum(1 for pid in train_person_ids if train_val_person_labels[pid] == 0)
    nc_count = sum(1 for pid in train_person_ids if train_val_person_labels[pid] == 1)
    print(f"Train AD: {ad_count}, NC: {nc_count}") # Train AD: 416, NC: 444

def check_patient_overlap(train_person_ids, test_person_ids):
    """Ensure there are no patients in both the test and train set"""
    train_set = set(train_person_ids)
    test_set = set(test_person_ids)
    overlap = train_set & test_set
    if overlap: # there are overlapping
        print("there are overlapping patient id's in the test and train set")
        print(f"overlap: {overlap}") 
        sys.exit(1) # model will not work well, so just exit

if CHECKS:
    check_normalisation(test_loader)
    check_classification_distirbution(train_person_ids, train_val_person_labels)
    visualise_image(train_loader)
    check_patient_overlap(train_person_ids, test_person_ids)
