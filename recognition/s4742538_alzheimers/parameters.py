"""
Holds general parameters for the project
"""

import os 

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

SAMPLE_IMAGE_FILENAME = "sample_image.png"

MODEL_FILENAME = "E20.pth"

MODEL_CONFIG = {
    "in_chans": 1,
    "num_classes": 2,
    "depths": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768]
}

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
