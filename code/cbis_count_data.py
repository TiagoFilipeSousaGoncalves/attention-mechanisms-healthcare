# Imports
import numpy as np
import _pickle as cPickle
import os
from PIL import Image

# Sklearn Import
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from model_utilities import DenseNet121, ResNet50, VGG16, MultiLevelDAM
from cbis_data_utilities import map_images_and_labels, TorchDatasetFromNumpyArray


# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBIS_proprocessed"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Results and Weights
weights_dir = os.path.join("results", "cbis", "weights")
if os.path.isdir(weights_dir) == False:
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join("results", "cbis", "history")
if os.path.isdir(history_dir) == False:
    os.makedirs(history_dir)


# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Choose Model Name
MODEL_NAME = 'ResNet50'
USE_ATTENTION = True


# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 224
WIDTH = 224

# Y dimensions
_, _, NR_CLASSES = map_images_and_labels(dir=train_dir)

# TODO: Review
# If we use Sigmoid activation
NR_CLASSES -= 1

# DenseNet121
if MODEL_NAME == 'DenseNet121':
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=MODEL_NAME.lower()
        )

    else:
        model = DenseNet121(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES
        )


# ResNet-50
elif MODEL_NAME == 'ResNet50':
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=MODEL_NAME.lower()
        )

    else:
        model = ResNet50(
            channels = CHANNELS,
            height = HEIGHT,
            width = WIDTH,
            nr_classes = NR_CLASSES
        )


# VGG-16
elif MODEL_NAME == "VGG16":
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=MODEL_NAME.lower()
        ) 

    else:
        model = VGG16(
            channels = CHANNELS,
            height = HEIGHT,
            width = WIDTH,
            nr_classes = NR_CLASSES
        )


# Hyper-parameters
EPOCHS = 300
# LOSS = torch.nn.CrossEntropyLoss()
LOSS = torch.nn.BCELoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = 2


# Load data
# Train Dataset
train_set = TorchDatasetFromNumpyArray(base_data_path=train_dir)
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")

# Validation
val_set = TorchDatasetFromNumpyArray(base_data_path=val_dir)
print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")

# Test
test_set = TorchDatasetFromNumpyArray(base_data_path=test_dir)
print(f"Number of Test Images: {len(test_set)} | Label Dict: {test_set.labels_dict}")