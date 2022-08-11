# Imports
import os
import argparse
import numpy as np

# Sklearn Import
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from model_utilities import DenseNet121, ResNet50, VGG16, MultiLevelDAM
from data_utilities import TorchDatasetFromNumpyArray, TorchDatasetFromPickle, get_transforms



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Choose the data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBIS", "MIMICCXR"], help="Choose the data set: CBIS, MIMICCXR.")

# Number of Classes
parser.add_argument('--nr_classes', type=int, default=1, help="Number of classes (using sigmoid, 1; using softmax, 2).")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The ID of the GPU device.")

# Batch size
parser.add_argument("--batchsize", type=int, default=1, help="Batch size for dataloaders.")


# Parse the arguments
args = parser.parse_args()


# Data directory
DATA_DIR = args.data_dir

# Data set
DATASET = args.dataset

# Backbone
BACKBONE = args.backbone

# Number of Classes
NR_CLASSES = args.nr_classes

# Batch size
BATCH_SIZE = args.batchsize


# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 224
WIDTH = 224

# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



# Transforms
test_transforms = get_transforms(split="test", data_aug=False, height=HEIGHT, width=WIDTH, mean=MEAN, std=STD)


# CBIS
if DATASET == "CBIS":

    # Directories
    test_dir = os.path.join(DATA_DIR, "test")

    # Dataset Objects
    test_set = TorchDatasetFromNumpyArray(base_data_path=test_dir, transform=test_transforms)


# MIMICCXR
elif DATASET == "MIMICCXR":

    # Directories
    test_dir = os.path.join(DATA_DIR, "Test_images_AP_resized")

    # Dataset Objects
    test_set = TorchDatasetFromPickle(base_data_path=test_dir, pickle_path=os.path.join(test_dir, "Annotations.pickle"), transform=test_transforms)



# DataLoaders
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)



# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"


# Results and Weights
weights_dir = os.path.join("results", DATASET.lower(), "weights")
if os.path.isdir(weights_dir) == False:
    os.makedirs(weights_dir)



# Lists to iterate
MODEL_NAMES = ['densenet121', 'resnet50', 'vgg16']
BEST_WEIGHTS = ['tr', 'val']
TRAIN_SETTINGS = ['baseline', 'baselinedaug', 'mldam', 'mldamdaug']
BASELINE = ['baseline', 'baselinedaug']
USE_ATTENTION = ['mldam', 'mldamdaug']



# Go through MODEL NAMES
for model_name in MODEL_NAMES:

    # Go through train and val weights
    for weight_split in BEST_WEIGHTS:

        # Go through train settings
        for train_setting in TRAIN_SETTINGS:

            # DenseNet121
            if model_name == 'densenet121':
                if train_setting in USE_ATTENTION:
                    model =  MultiLevelDAM(
                        channels=CHANNELS,
                        height=HEIGHT,
                        width=WIDTH,
                        nr_classes=NR_CLASSES,
                        backbone=model_name.lower()
                    )

                else:
                    model = DenseNet121(
                        channels=CHANNELS,
                        height=HEIGHT,
                        width=WIDTH,
                        nr_classes=NR_CLASSES
                    )


            # ResNet-50
            elif model_name == 'resnet50':
                if train_setting in USE_ATTENTION:
                    model =  MultiLevelDAM(
                        channels=CHANNELS,
                        height=HEIGHT,
                        width=WIDTH,
                        nr_classes=NR_CLASSES,
                        backbone=model_name.lower()
                    )

                else:
                    model = ResNet50(
                        channels = CHANNELS,
                        height = HEIGHT,
                        width = WIDTH,
                        nr_classes = NR_CLASSES
                    )


            # VGG-16
            elif model_name == "vgg16":
                if train_setting in USE_ATTENTION:
                    model =  MultiLevelDAM(
                        channels=CHANNELS,
                        height=HEIGHT,
                        width=WIDTH,
                        nr_classes=NR_CLASSES,
                        backbone=model_name.lower()
                    ) 

                else:
                    model = VGG16(
                        channels = CHANNELS,
                        height = HEIGHT,
                        width = WIDTH,
                        nr_classes = NR_CLASSES
                    )



            # Initialise lists to compute scores
            y_test_true = list()
            y_test_pred = list()


            try:
                # Load weights and put model in evaluation mode
                model = model.to(DEVICE)

                # Get final model name
                if train_setting in BASELINE:
                    if train_setting == "baseline":
                        weights_file = f"{model_name.lower()}_{weight_split}_{DATASET.lower()}.pt"
                    else:
                        weights_file = f"{model_name.lower()}daug_{weight_split}_{DATASET.lower()}.pt"
                
                elif train_setting in USE_ATTENTION:
                    if train_setting == "mldam":
                        weights_file = f"{model_name.lower()}_mldam_{weight_split}_{DATASET.lower()}.pt"
                    else:
                        weights_file = f"{model_name.lower()}_mldamdaug_{weight_split}_{DATASET.lower()}.pt"


                weights_fname = os.path.join(weights_dir, weights_file)
                model.load_state_dict(torch.load(weights_fname, map_location=DEVICE))
                model.eval()
                

                # Deactivate gradients
                with torch.no_grad():

                    # Iterate through dataloader
                    for batch_idx, (images, labels) in enumerate(test_loader):

                        # Move data data anda model to GPU (or not)
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        

                        # Forward pass: compute predicted outputs by passing inputs to the model
                        logits = model(images)

                        # Concatenate lists
                        y_test_true += list(labels.cpu().detach().numpy())

                        # Using Sigmoid Activation (we apply a threshold of 0.5 in probabilities)
                        y_test_pred += list(logits.cpu().detach().numpy())
                        y_test_pred = [1 if i >= 0.5 else 0 for i in y_test_pred]


                    # Compute Training Accuracy
                    test_acc = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
                    test_recall = recall_score(y_true=y_test_true, y_pred=y_test_pred)
                    test_precision = precision_score(y_true=y_test_true, y_pred=y_test_pred)
                    test_f1 = f1_score(y_true=y_test_true, y_pred=y_test_pred)


                    # Print Statistics
                    print(f'Model Name: {model_name} | Weight Split: {weight_split} | Train Setting: {train_setting}')
                    print(f"Accuracy: {test_acc} | Recall: {test_recall} | Precision: {test_precision} | F1-Score: {test_f1}")
                    print('\n')

            except:
                print(f'Model Name: {model_name} | Weight Split: {weight_split} | Train Setting: {train_setting}')
                print('Error')
                print('\n')



print("Finished.")
