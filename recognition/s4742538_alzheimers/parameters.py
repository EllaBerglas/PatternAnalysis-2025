"""
Holds general parameters for the project
"""

DATA_ROOT = "./data/AD_NC/"
MODEL_SIZE = "SMALL" # "SMALL", "TINY", "BASE"

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 100
COMPILE = True
AUGMENTED_DS = True
NUM_CLASSES = 1 # 1 for bce and 2 for cross entropy
AUG_PROB = 1
CHECKS = True
THRESHOLD = 0.7 # evaluation threshold

MODEL_FILENAME = f"./model/alzhiemers_classification_model.pth"

MODEL_CONFIG= { # small
    "in_chans": 20,
    "num_classes":NUM_CLASSES,
    "depths": [3, 3, 27, 3],
    "dims": [96, 192, 384, 768],
    "drop_path_rate": 0.2  # dropout for residual connections
}


RANDOM_STATE = 42
#NORMALISATION_M = 0.1114
#NORMALISATION_SD = 0.2184
NORMALISATION_M = 0.5
NORMALISATION_SD = 0.5
CHANNELS = 1 # per image (greyscale)
IMAGE_SIZE = 224

import os 

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")



