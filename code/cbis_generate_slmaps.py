# Imports
# General Imports
import numpy as np
import os
import _pickle as cPickle
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision

# Torchsummary
from torchsummary import summary

# Captum
from captum.attr import visualization as viz
from captum.attr import DeepLift

# Project Imports
from model_utilities import DenseNet121, ResNet50, VGG16, MultiLevelDAM
from cbis_data_utilities import map_images_and_labels, TorchDatasetFromNumpyArray

# To make computations deterministic, let's fix random seeds:
torch.manual_seed(42)
np.random.seed(42)



# A generic function that will be used for calling attribute on attribution algorithm defined in input
def attribute_image_features(model, algorithm, img_input, gt_label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(img_input, target=gt_label, **kwargs)
    
    return tensor_attributions



# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBIS_proprocessed"
test_dir = os.path.join(data_dir, "test")
weights_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "cbis", "weights")
results_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "cbis")


# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"




# Lists to iterate
MODEL_NAMES = ['DenseNet121', 'ResNet50', 'VGG16']
BEST_WEIGHTS = ['tr', 'val']
TRAIN_SETTINGS = ['baseline', 'baselinedaug', 'mldam', 'mldamdaug']
USE_ATTENTION = ['mldam', 'mldamdaug']


# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 224
WIDTH = 224

# Y dimensions
_, _, NR_CLASSES = map_images_and_labels(dir=test_dir)


# If we use Sigmoid activation
NR_CLASSES -= 1


# Load data
# Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Dataset
test_set = TorchDatasetFromNumpyArray(base_data_path=test_dir, transform=test_transforms)

# Train Dataloader
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)


# Go through MODEL NAMES
for model_name in MODEL_NAMES:

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




# Go through test data
y_true, y_pred = list(), list()
for batch_idx, batch_data in enumerate(test_loader):
    print(f"Batch: {batch_idx}")
    # Get data
    images, labels = batch_data

    # Get logits
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)

    # Original Image
    for index in range(images.size(0)):
        original_image = np.transpose((images[index].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

        # Input to the xAI models
        input_img = images[index].unsqueeze(0)
        input_img.requires_grad = True


        # DeepLift
        dl = DeepLift(model)
        attr_dl = attribute_image_features(model, dl, input_img, labels[index])
        # print(attr_dl.size())
        attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))



        
        # print('Original Image')
        print('Label: {} | Predicted: {}'.format(labels[index], predictions[index]))

        original_image = np.transpose((images[index].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

        # Show Original Image
        _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")
        save = input(f"Save original image NumPy array (Y/N)? ")
        if save.lower() == "y":
            name = input(f"Enter the name for the NumPy array: ")
            name = name + ".npy"
            save_path = os.path.join(results_dir, figures_dir, dataset_figs_dir, "Original", name)
            np.save(file=save_path, arr=original_image, allow_pickle=True)

        


        # DeepLift
        _ = viz.visualize_image_attr(attr_dl, None, method="heat_map", sign="all", show_colorbar=True, title="Heatmap DeepLift")
        _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True, title="Overlayed DeepLift")
        save = input(f"Save deeplift attribution NumPy array (Y/N)? ")
        if save.lower() == "y":
            name = input(f"Enter the name for the NumPy array: ")
            name = name + ".npy"
            save_path = os.path.join(results_dir, figures_dir, dataset_figs_dir, "DeepLift", name)
            np.save(file=save_path, arr=attr_dl, allow_pickle=True)



print("Finished.")