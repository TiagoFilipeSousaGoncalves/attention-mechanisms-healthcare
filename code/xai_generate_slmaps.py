# Imports
import os
import argparse
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Captum Imports
from captum.attr import Saliency

# Project Imports
from model_utilities import DenseNet121, ResNet50, VGG16, MultiLevelDAM, attribute_image_features
from data_utilities import TorchDatasetFromPickle, TorchDatasetFromNumpyArray, get_transforms, unnormalize

# To make computations deterministic, let's fix random seeds:
torch.manual_seed(42)
np.random.seed(42)



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



# Results, Weights and Attributes
results_dir = os.path.join("results", DATASET.lower())
weights_dir = os.path.join("results", DATASET.lower(), "weights")
attributes_dir = os.path.join(results_dir, "attributes")
if not os.path.isdir(attributes_dir):
    os.makedirs(attributes_dir)



# Lists to iterate
MODEL_NAMES = ['DenseNet121', 'ResNet50', 'VGG16']
BEST_WEIGHTS = ['tr', 'val']
TRAIN_SETTINGS = ['baseline', 'baselinedaug', 'mldam', 'mldamdaug']
USE_ATTENTION = ['mldam', 'mldamdaug']



# Go through MODEL NAMES
for model_name in MODEL_NAMES:

    if os.path.isdir(os.path.join(attributes_dir, f"{model_name.lower()}")) == False:
        os.makedirs(os.path.join(attributes_dir, f"{model_name.lower()}"))

    # Iterate through dataloader
    for batch_idx, (images, labels) in enumerate(test_loader):

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE), labels.to(DEVICE)


        if model_name == 'DenseNet121':
            model = DenseNet121(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)
            mldam_model = MultiLevelDAM(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES, backbone=model_name.lower())

        
        elif model_name == 'ResNet50':
            model = ResNet50(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)
            mldam_model = MultiLevelDAM(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES, backbone=model_name.lower())


        elif model_name == 'VGG16':
            model = VGG16(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)
            mldam_model = MultiLevelDAM(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES, backbone=model_name.lower())





        # Load models to assess logits
        # Baseline
        baseline_weights = os.path.join(weights_dir, f"{model_name.lower()}_val_{DATASET.lower()}.pt")
        model.to(DEVICE)
        model.load_state_dict(torch.load(baseline_weights, map_location=DEVICE))
        model.eval()

        # Get prediction
        baseline_pred = model(images)
        baseline_pred = 1 if baseline_pred[0].item() >= 0.5 else 0



        # Baseline Data Augmentation
        baselinedaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}daug_val_{DATASET.lower()}.pt")
        model.to(DEVICE)
        model.load_state_dict(torch.load(baselinedaug_weigths, map_location=DEVICE))
        model.eval()

        # Get prediction
        baselinedaug_pred = model(images)
        baselinedaug_pred = 1 if baselinedaug_pred[0].item() >= 0.5 else 0

        
        # MLDAM
        mldam_weights = os.path.join(weights_dir, f"{model_name.lower()}_mldam_val_{DATASET.lower()}.pt")
        mldam_model.to(DEVICE)
        mldam_model.load_state_dict(torch.load(mldam_weights, map_location=DEVICE))
        mldam_model.eval()

        # Get prediction
        mldam_pred = mldam_model(images)
        mldam_pred = 1 if mldam_pred[0].item() >= 0.5 else 0


        
        # MLDAM w/ Data Augmentation
        mldamdaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_mldamdaug_val_{DATASET.lower()}.pt")
        mldam_model.to(DEVICE)
        mldam_model.load_state_dict(torch.load(mldamdaug_weigths, map_location=DEVICE))
        mldam_model.eval()

        # Get prediction
        mldamdaug_pred = mldam_model(images)
        mldamdaug_pred = 1 if mldamdaug_pred[0].item() >= 0.5 else 0

        
        
        # Labels
        label = int(labels[0].item())


        # Check if these correspond to the correct label
        if baseline_pred == label and baselinedaug_pred == label and mldam_pred == label and mldamdaug_pred == label:
            original_image = np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
            original_image = unnormalize(original_image, MEAN, STD)

            # Save image
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_original_{label}.npy"),
                arr=original_image,
                allow_pickle=True
            )



            # Input to the xAI models
            input_img = images[0].unsqueeze(0)
            input_img.requires_grad = True


            # DeepLift - Baseline
            baseline_weights = os.path.join(weights_dir, f"{model_name.lower()}_val_{DATASET.lower()}.pt")
            model.to(DEVICE)
            model.load_state_dict(torch.load(baseline_weights, map_location=DEVICE))
            model.eval()

            dl_baseline = Saliency(model)
            attr_dl_baseline = attribute_image_features(model, dl_baseline, input_img, None, abs=False)
            attr_dl_baseline = np.transpose(attr_dl_baseline.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_baseline_label_{label}.npy"),
                arr=attr_dl_baseline,
                allow_pickle=True
            )



            # DeepLift - Baseline Data Augmentation
            baselinedaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}daug_val_{DATASET.lower()}.pt")
            model.to(DEVICE)
            model.load_state_dict(torch.load(baselinedaug_weigths, map_location=DEVICE))
            model.eval()

            dl_baselinedaug = Saliency(model)
            attr_dl_baselinedaug = attribute_image_features(model, dl_baselinedaug, input_img, None, abs=False)
            attr_dl_baselinedaug = np.transpose(attr_dl_baselinedaug.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_baselinedaug_label_{label}.npy"),
                arr=attr_dl_baselinedaug,
                allow_pickle=True
            )



            # MLDAM
            mldam_weights = os.path.join(weights_dir, f"{model_name.lower()}_mldam_val_{DATASET.lower()}.pt")
            mldam_model.to(DEVICE)
            mldam_model.load_state_dict(torch.load(mldam_weights, map_location=DEVICE))
            mldam_model.eval()

            dl_mldam = Saliency(mldam_model)
            attr_dl_mldam = attribute_image_features(mldam_model, dl_mldam, input_img, None, abs=False)
            attr_dl_mldam = np.transpose(attr_dl_mldam.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_mldam_label_{label}.npy"),
                arr=attr_dl_mldam,
                allow_pickle=True
            )



            # MLDAM w/ Data Augmentation
            mldamdaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_mldamdaug_val_{DATASET.lower()}.pt")
            mldam_model.to(DEVICE)
            mldam_model.load_state_dict(torch.load(mldamdaug_weigths, map_location=DEVICE))
            mldam_model.eval()

            dl_mldamdaug = Saliency(mldam_model)
            attr_dl_mldamdaug = attribute_image_features(mldam_model, dl_mldamdaug, input_img, None, abs=False)
            attr_dl_mldamdaug = np.transpose(attr_dl_mldamdaug.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_mldamdaug_label_{label}.npy"),
                arr=attr_dl_mldamdaug,
                allow_pickle=True
            )



print("Finished.")
