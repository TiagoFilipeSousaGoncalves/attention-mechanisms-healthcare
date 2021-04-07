# Imports
import numpy as np
import _pickle as cPickle
import os


# PyTorch Imports
import torch
import torchvision


# Project Imports
from .model_utilities import DenseNet121
from .cbis_data_utilities import map_images_and_labels



# Prepare data
# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBIS_proprocessed"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# Load data
# Train
train_set = 
train_transforms = 
train_loader = 

# Validation
val_set = 
val_transforms = 
val_loader = 


# Data dimensions
CHANNELS = 
HEIGHT = 
WIDTH = 

# Target dimensions
NR_CLASSES = 


# Choose Model Name
MODEL_NAME = 'DenseNet121'

# Create Model instance based on name
if MODEL_NAME == 'DenseNet121':
    model = DenseNet121(
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        nr_classes=NR_CLASSES
    )


# Hyper-parameters
EPOCHS = 300
LOSS = torch.nn.BCELoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters, lr=LEARNING_RATE)


# Training Loop
