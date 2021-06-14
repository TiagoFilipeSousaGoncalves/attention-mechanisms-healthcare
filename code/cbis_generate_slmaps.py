# Imports
import numpy as np
import os

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Captum Imports
from captum.attr import Saliency

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



# Function to unnormalize images
def unnormalize(image, mean_array, std_array):

    # Create a copy
    unnormalized_img = image.copy()

    # Get channels
    _, _, channels = unnormalized_img.shape


    for c in range(channels):
        unnormalized_img[:, :, c] = image[:, :, c] * std_array[c] + mean_array[c]


    return unnormalized_img



# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBIS_proprocessed"
test_dir = os.path.join(data_dir, "test")
weights_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "cbis", "weights")
results_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "cbis")
attributes_dir = os.path.join(results_dir, "attributes")
if os.path.isdir(attributes_dir) == False:
    os.makedirs(attributes_dir)


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
        baseline_weights = os.path.join(weights_dir, f"{model_name.lower()}_baseline_val_cbis.pt")
        model.to(DEVICE)
        model.load_state_dict(torch.load(baseline_weights, map_location=DEVICE))
        model.eval()

        # Get prediction
        baseline_pred = model(images)
        baseline_pred = 1 if baseline_pred[0].item() >= 0.5 else 0



        # Baseline Data Augmentation
        baselinedaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_baselinedaug_val_cbis.pt")
        model.to(DEVICE)
        model.load_state_dict(torch.load(baselinedaug_weigths, map_location=DEVICE))
        model.eval()

        # Get prediction
        baselinedaug_pred = model(images)
        baselinedaug_pred = 1 if baselinedaug_pred[0].item() >= 0.5 else 0

        
        # MLDAM
        mldam_weights = os.path.join(weights_dir, f"{model_name.lower()}_mldam_val_cbis.pt")
        mldam_model.to(DEVICE)
        mldam_model.load_state_dict(torch.load(mldam_weights, map_location=DEVICE))
        mldam_model.eval()

        # Get prediction
        mldam_pred = mldam_model(images)
        mldam_pred = 1 if mldam_pred[0].item() >= 0.5 else 0


        
        # MLDAM w/ Data Augmentation
        mldamdaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_mldamdaug_val_cbis.pt")
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
            baseline_weights = os.path.join(weights_dir, f"{model_name.lower()}_baseline_val_cbis.pt")
            model.to(DEVICE)
            model.load_state_dict(torch.load(baseline_weights, map_location=DEVICE))
            model.eval()

            dl_baseline = Saliency(model)
            attr_dl_baseline = attribute_image_features(model, dl_baseline, input_img, labels[0], abs=False)
            attr_dl_baseline = np.transpose(attr_dl_baseline.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_baseline_label_{label}.npy"),
                arr=attr_dl_baseline,
                allow_pickle=True
            )



            # DeepLift - Baseline Data Augmentation
            baselinedaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_baselinedaug_val_cbis.pt")
            model.to(DEVICE)
            model.load_state_dict(torch.load(baselinedaug_weigths, map_location=DEVICE))
            model.eval()

            dl_baselinedaug = Saliency(model)
            attr_dl_baselinedaug = attribute_image_features(model, dl_baselinedaug, input_img, labels[0], abs=False)
            attr_dl_baselinedaug = np.transpose(attr_dl_baselinedaug.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_baselinedaug_label_{label}.npy"),
                arr=attr_dl_baselinedaug,
                allow_pickle=True
            )



            # MLDAM
            mldam_weights = os.path.join(weights_dir, f"{model_name.lower()}_mldam_val_cbis.pt")
            mldam_model.to(DEVICE)
            mldam_model.load_state_dict(torch.load(mldam_weights, map_location=DEVICE))
            mldam_model.eval()

            dl_mldam = Saliency(mldam_model)
            attr_dl_mldam = attribute_image_features(mldam_model, dl_mldam, input_img, labels[0], abs=False)
            attr_dl_mldam = np.transpose(attr_dl_mldam.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_mldam_label_{label}.npy"),
                arr=attr_dl_mldam,
                allow_pickle=True
            )



            # MLDAM w/ Data Augmentation
            mldamdaug_weigths = os.path.join(weights_dir, f"{model_name.lower()}_mldamdaug_val_cbis.pt")
            mldam_model.to(DEVICE)
            mldam_model.load_state_dict(torch.load(mldamdaug_weigths, map_location=DEVICE))
            mldam_model.eval()

            dl_mldamdaug = Saliency(mldam_model)
            attr_dl_mldamdaug = attribute_image_features(mldam_model, dl_mldamdaug, input_img, labels[0], abs=False)
            attr_dl_mldamdaug = np.transpose(attr_dl_mldamdaug.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

            # Save attribute
            np.save(
                file=os.path.join(attributes_dir, f"{model_name.lower()}", f"img_{batch_idx}_dl_mldamdaug_label_{label}.npy"),
                arr=attr_dl_mldamdaug,
                allow_pickle=True
            )



print("Finished.")