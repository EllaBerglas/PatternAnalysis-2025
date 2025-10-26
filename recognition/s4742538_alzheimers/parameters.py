"""
Holds general parameters for the project
"""

# from google.colab import drive
# drive.mount('/content/gdrive')
# DATA_ROOT = "./gdrive/My Drive/Colab Notebooks/data/AD_NC/"

DATA_ROOT = "./data/AD_NC/"

CHANNELS = 1 # per image
IMAGE_SIZE = 384
NORMALISATION_M = 0.1114
NORMALISATION_SD = 0.2184
BATCH_SIZE = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 15

MODEL_FILENAME = f"20Chan_{EPOCHS}_tiny_transform.pth"

MODEL_CONFIG = { # tiny
    "in_chans": 20,
    "num_classes": 2,
    "depths": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768]
}

# MODEL_CONFIG = { # small
#     "in_chans": 20,
#     "num_classes": 2,
#     "depths": [3, 3, 27, 3],
#     "dims": [96, 192, 384, 768],
#     "drop_path_rate": 0.2  # dropout for residual connections
# }

# MODEL_CONFIG = { # base
#     "in_chans": 20,
#     "num_classes": 2,
#     "depths": [3, 3, 27, 3],
#     "dims": [128, 256, 512, 1024]
# }

import os 

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
# TRAIN_AD = os.path.join(TRAIN_DIR, "AD")
# TRAIN_NC = os.path.join(TRAIN_DIR, "NC")

TEST_DIR = os.path.join(DATA_ROOT, "test")
# use these for specific evaluations??
# TEST_AD = os.path.join(TEST_DIR, "AD")
# TEST_NC = os.path.join(TEST_DIR, "NC")

SAMPLE_IMAGE_FILENAME = "sample_image.png"


