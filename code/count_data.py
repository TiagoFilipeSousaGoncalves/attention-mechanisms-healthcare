# Imports
import os
import argparse

# Project Imports
from data_utilities import TorchDatasetFromPickle, TorchDatasetFromNumpyArray



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Choose the data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBIS", "MIMICCXR"], help="Choose the data set: CBIS, MIMICCXR.")



# Parse the arguments
args = parser.parse_args()


# Data directory
DATA_DIR = args.data_dir

# Data set
DATASET = args.dataset


# CBIS
if DATASET == "CBIS":

    # Directories
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Dataset Objects
    train_set = TorchDatasetFromNumpyArray(base_data_path=train_dir)
    val_set = TorchDatasetFromNumpyArray(base_data_path=val_dir)
    test_set = TorchDatasetFromNumpyArray(base_data_path=test_dir)


# MIMICCXR
elif DATASET == "MIMICCXR":

    # Directories
    train_dir = os.path.join(DATA_DIR, "Train_images_AP_resized")
    val_dir = os.path.join(DATA_DIR, "Val_images_AP_resized")
    test_dir = os.path.join(DATA_DIR, "Test_images_AP_resized")

    # Dataset Objects
    train_set = TorchDatasetFromPickle(base_data_path = train_dir, pickle_path = os.path.join(train_dir, "Annotations.pickle"))
    val_set = TorchDatasetFromPickle(base_data_path = val_dir, pickle_path = os.path.join(val_dir, "Annotations.pickle"))
    test_set = TorchDatasetFromPickle(base_data_path=test_dir, pickle_path=os.path.join(test_dir, "Annotations.pickle"))



# Count data
print(f"Data set: {DATASET}")
print(f"Number of Train Images: {len(train_set)} | Label Dict: {train_set.labels_dict}")
print(f"Number of Validation Images: {len(val_set)} | Label Dict: {val_set.labels_dict}")
print(f"Number of Test Images: {len(test_set)} | Label Dict: {test_set.labels_dict}")
