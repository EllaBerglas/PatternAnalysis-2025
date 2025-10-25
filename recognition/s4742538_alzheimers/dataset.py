"""
Containing the data loader for loading and preprocessing your data
"""
from torchvision import transforms, datasets # type: ignore
from torch.utils.data import DataLoader, random_split, Subset # type: ignore
import matplotlib # type: ignore 
matplotlib.use("Agg") # to work in wsl
import matplotlib.pyplot as plt  # type: ignore
import numpy as np # type: ignore
from parameters import TRAIN_DIR, TEST_DIR, CHANNELS, IMAGE_SIZE, BATCH_SIZE, SAMPLE_IMAGE_FILENAME
import os
from sklearn.model_selection import train_test_split # type: ignore 
from collections import defaultdict 


# reduce image size, convert to tensor, and then normalise
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.Grayscale(num_output_channels=CHANNELS), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])  # pixel values [-1, 1]
])

# Get Images
train_val_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_data = datasets.ImageFolder(root=TEST_DIR, transform=transform)


# extrac the person_ids
image_paths = [path for path, _ in train_val_data.samples]
person_ids = [os.path.basename(path).split("_")[0] for path in image_paths]

person_to_indices = defaultdict(list) 
for idx, pid in enumerate(person_ids):
    person_to_indices[pid].append(idx) # stores (persons_id, [list of index's for that persons images])

unique_person_ids = list(person_to_indices.keys())

train_person_ids, val_person_ids = train_test_split(
    unique_person_ids, 
    test_size=0.2, 
    random_state=42 
)# better than the other one

# make indicies and their datasets for tran and val based on peopleid
train_indices = [idx for pid in train_person_ids for idx in person_to_indices[pid]]
val_indices = [idx for pid in val_person_ids for idx in person_to_indices[pid]]

train_dataset = Subset(train_val_data, train_indices)
val_dataset = Subset(train_val_data, val_indices)

# sanity check
print(f"train size {len(train_dataset)}")
print(f"val size {len(val_dataset)}")
print(f"test size {len(test_data)}")
print(f"unique persons {len(unique_person_ids)}")
print(f"train persons {len(train_person_ids)}")
print(f"val persons {len(val_person_ids)}")

# TODO: turn this to be error checking
# # I want to make the assumption that each person has 20 images.
# # yep this holds, for both train and test
# counts = {pid: len(indices) for pid, indices in person_to_indices.items()}
# avg_segments = sum(counts.values()) / len(counts)
# unique_counts = set(counts.values())

# print(f"Average segments per person: {avg_segments:.2f}")
# print(f"Unique segment counts: {unique_counts}")

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
plt.savefig(SAMPLE_IMAGE_FILENAME)
print(f"Label {label}")