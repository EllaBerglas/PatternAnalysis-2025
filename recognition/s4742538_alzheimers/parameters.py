"""
Holds general parameters for the project
"""
# uncomment fo ruse on colab
# from google.colab import drive
# drive.mount('/content/gdrive')
# DATA_ROOT = "./gdrive/My Drive/Colab Notebooks/data/AD_NC/"

DATA_ROOT = "./data/AD_NC/"
MODEL_SIZE = "SMALL" # "SMALL", "TINY", "BASE"

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 120
COMPILE = True
AUGMENTED_DS = True
# NUM_CLASSES = 2 # for cross entropy loss only
NUM_CLASSES = 1
AUG_PROB = 1
CHECKS = True

#MODEL_FILENAME = f"./models/100E_1_smoothing/20Chan_{EPOCHS}_{MODEL_SIZE}_E77_07266.pth"
MODEL_FILENAME = f"./final_runs/100E_20C_BCE_augment_clip_{EPOCHS}.pth"

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

SAMPLE_IMAGE_FILENAME = "sample_image.png"


