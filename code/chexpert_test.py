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
test_dir = os.path.join(data_dir, "Test_images_AP_resized")
weights_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "chexpert", "weights")
results_dir = os.path.join("/ctm-hdd-pool01/tgoncalv/attention-mechanisms-healthcare", "results", "chexpert")



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
NR_CLASSES = 1


# Go through MODEL NAMES
for model_name in MODEL_NAMES:

    # Go through train and val weights
    for weight_split in BEST_WEIGHTS:

        # Go through train settings
        for train_setting in TRAIN_SETTINGS:

            # DenseNet121
            if model_name == 'DenseNet121':
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
            elif model_name == 'ResNet50':
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
            elif model_name == "VGG16":
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


            # Hyper-parameters
            BATCH_SIZE = 1


            # Load data
            # Transforms
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD)
            ])

            # Train Dataset
            test_set = TorchDatasetFromPickle(base_data_path=test_dir, pickle_path=os.path.join(test_dir, "Annotations.pickle"), transform=test_transforms)

            # Train Dataloader
            test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)



            # Initialise lists to compute scores
            y_test_true = list()
            y_test_pred = list()


            try:
                # Load weights and put model in evaluation mode
                model = model.to(DEVICE)
                weights_fname = os.path.join(weights_dir, f"{model_name.lower()}_{train_setting}_{weight_split}_chexpert.pt")
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
                    print(f'\n')

            except:
                print(f'Model Name: {model_name} | Weight Split: {weight_split} | Train Setting: {train_setting}')
                # print(f"Accuracy: {test_acc} | Recall: {test_recall} | Precision: {test_precision} | F1-Score: {test_f1}")
                print('Error')
                print(f'\n')

                


# Finish statement
print("Finished.")