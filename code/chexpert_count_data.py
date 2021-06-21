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
from chexpert_data_utilities import imgs_and_labels_from_pickle, TorchDatasetFromPickle


# Directories
data_dir = "/ctm-hdd-pool01/wjsilva19/MedIA"
train_dir = os.path.join(data_dir, "Train_images_AP_resized")
val_dir = os.path.join(data_dir, "Val_images_AP_resized")
test_dir = os.path.join(data_dir, "Test_images_AP_resized")



# Load data
# Train Dataset
train_set = TorchDatasetFromPickle(base_data_path = train_dir, pickle_path = os.path.join(train_dir, "Annotations.pickle"))
print(f"Number of Train Images: {len(train_set)}")

# Validation Dataset
val_set = TorchDatasetFromPickle(base_data_path = val_dir, pickle_path = os.path.join(val_dir, "Annotations.pickle"))
print(f"Number of Validation Images: {len(val_set)}")

# Test Dataset
test_set = TorchDatasetFromPickle(base_data_path=test_dir, pickle_path=os.path.join(test_dir, "Annotations.pickle"))
print(f"Number of Test Images: {len(test_set)}")