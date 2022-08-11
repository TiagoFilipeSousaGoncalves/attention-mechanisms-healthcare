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

# Choose model (backbone)
parser.add_argument('--backbone', type=str, required=True, choices=["densenet121", "resnet50", "vgg16"], help="Choose a backbone: densenet121, resnet50, vgg16.")

# Use MLDAM (attention)
parser.add_argument('--use_attention', action="store_true", help="Use MLDAM (attention).")

# Number of Classes
parser.add_argument('--nr_classes', type=int, default=1, help="Number of classes (using sigmoid, 1; using softmax, 2).")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The ID of the GPU device.")

# Batch size
parser.add_argument("--batchsize", type=int, default=2, help="Batch size for dataloaders.")

# Epochs
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")

# Data Augmentation
parser.add_argument('--use_daug', action="store_true", help="Use data augmentation.")


# Parse the arguments
args = parser.parse_args()


# Data directory
DATA_DIR = args.data_dir

# Data set
DATASET = args.dataset

# Backbone
BACKBONE = args.backbone

# MLDAM (Attention)
USE_ATTENTION = args.use_attention

# DAUG (Data Augmentation)
USE_DAUG = args.use_daug

# Number of Classes
NR_CLASSES = args.nr_classes


# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 224
WIDTH = 224



# DenseNet121
if BACKBONE == 'densenet121':
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=BACKBONE
        )

        MODEL_NAME = f"{BACKBONE}_mldam"

    else:
        model = DenseNet121(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES
        )

        MODEL_NAME = BACKBONE


# ResNet-50
elif BACKBONE == 'ResNet50':
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=BACKBONE
        )

        MODEL_NAME = f"{BACKBONE}_mldam"

    else:
        model = ResNet50(
            channels = CHANNELS,
            height = HEIGHT,
            width = WIDTH,
            nr_classes = NR_CLASSES
        )

        MODEL_NAME = BACKBONE


# VGG-16
elif BACKBONE == "VGG16":
    if USE_ATTENTION:
        model =  MultiLevelDAM(
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            nr_classes=NR_CLASSES,
            backbone=BACKBONE
        )

        MODEL_NAME = f"{BACKBONE}_mldam"

    else:
        model = VGG16(
            channels = CHANNELS,
            height = HEIGHT,
            width = WIDTH,
            nr_classes = NR_CLASSES
        )

        MODEL_NAME = BACKBONE



# Results and Weights
weights_dir = os.path.join("results", DATASET.lower(), "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join("results", DATASET.lower(), "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"



# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Hyper-parameters
EPOCHS = args.epochs
LOSS = torch.nn.BCELoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = args.batchsize


# Transforms
# Train
if USE_DAUG:
    train_transforms = get_transforms(split="train", data_aug=True, height=HEIGHT, width=WIDTH, mean=MEAN, std=STD)
    MODEL_NAME = f"{MODEL_NAME}daug"

else:
    train_transforms = get_transforms(split="train", data_aug=False, height=HEIGHT, width=WIDTH, mean=MEAN, std=STD)


# Validation
val_transforms = get_transforms(split="validation", data_aug=False, height=HEIGHT, width=WIDTH, mean=MEAN, std=STD)


# CBIS
if DATASET == "CBIS":

    # Directories
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    # test_dir = os.path.join(DATA_DIR, "test")

    # Dataset Objects
    train_set = TorchDatasetFromNumpyArray(base_data_path=train_dir, transform=train_transforms)
    val_set = TorchDatasetFromNumpyArray(base_data_path=val_dir, transform=val_transforms)
    # test_set = TorchDatasetFromNumpyArray(base_data_path=test_dir)


# MIMICCXR
elif DATASET == "MIMICCXR":

    # Directories
    train_dir = os.path.join(DATA_DIR, "Train_images_AP_resized")
    val_dir = os.path.join(DATA_DIR, "Val_images_AP_resized")
    test_dir = os.path.join(DATA_DIR, "Test_images_AP_resized")

    # Dataset Objects
    train_set = TorchDatasetFromPickle(base_data_path=train_dir, pickle_path=os.path.join(train_dir, "Annotations.pickle"), transform=train_transforms)
    val_set = TorchDatasetFromPickle(base_data_path=val_dir, pickle_path=os.path.join(val_dir, "Annotations.pickle"), transform=val_transforms)
    # test_set = TorchDatasetFromPickle(base_data_path=test_dir, pickle_path=os.path.join(test_dir, "Annotations.pickle"))



# DataLoaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((EPOCHS, 4))
val_metrics = np.zeros_like(train_metrics)


# Go through the number of Epochs
for epoch in range(EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = list()
    y_train_pred = list()


    # Running train loss
    run_train_loss = 0.0


    # Put model in training mode
    model.train()


    # Iterate through dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model = model.to(DEVICE)


        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad()


        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)
        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        # loss = LOSS(logits, labels)

        # Using BCE w/ Sigmoid
        loss = LOSS(logits.reshape(-1).float(), labels.float())
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()
        
        # Update batch losses
        run_train_loss += (loss.item() * images.size(0))

        # Concatenate lists
        y_train_true += list(labels.cpu().detach().numpy())
        
        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        # s_logits = torch.nn.Softmax(dim=1)(logits)
        # s_logits = torch.argmax(s_logits, dim=1)
        # y_train_pred += list(s_logits.cpu().detach().numpy())

        # Using Sigmoid Activation (we apply a threshold of 0.5 in probabilities)
        y_train_pred += list(logits.cpu().detach().numpy())
        y_train_pred = [1 if i >= 0.5 else 0 for i in y_train_pred]


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)

    # Compute Train Metrics
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred)
    train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred)
    train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    # Save it to directory
    np.save(
        file=os.path.join(history_dir, f"{MODEL_NAME.lower()}_tr_{DATASET.lower()}_losses.npy"),
        arr=train_losses,
        allow_pickle=True
    )


    # Train Metrics
    # Acc
    train_metrics[epoch, 0] = train_acc
    # Recall
    train_metrics[epoch, 1] = train_recall
    # Precision
    train_metrics[epoch, 2] = train_precision
    # F1-Score
    train_metrics[epoch, 3] = train_f1
    # Save it to directory
    np.save(
        file=os.path.join(history_dir, f"{MODEL_NAME.lower()}_tr_{DATASET.lower()}_metrics.npy"),
        arr=train_metrics,
        allow_pickle=True
    )


    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss

        # Save checkpoint
        model_path = os.path.join(weights_dir, f"{MODEL_NAME.lower()}_tr_{DATASET.lower()}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Successfully saved at: {model_path}")





    # Validation Loop
    print("Validation Phase")


    # Initialise lists to compute scores
    y_val_true = list()
    y_val_pred = list()


    # Running train loss
    run_val_loss = 0.0


    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for batch_idx, (images, labels) in enumerate(val_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            # loss = LOSS(logits, labels)

            # Using BCE w/ Sigmoid
            loss = LOSS(logits.reshape(-1).float(), labels.float())
            
            # Update batch losses
            run_val_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_val_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            # s_logits = torch.nn.Softmax(dim=1)(logits)
            # s_logits = torch.argmax(s_logits, dim=1)
            # y_val_pred += list(s_logits.cpu().detach().numpy())

            # Using Sigmoid Activation (we apply a threshold of 0.5 in probabilities)
            y_val_pred += list(logits.cpu().detach().numpy())
            y_val_pred = [1 if i >= 0.5 else 0 for i in y_val_pred]

        

        # Compute Average Train Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Training Accuracy
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred)
        val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred)
        val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred)

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Train Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        np.save(
            file=os.path.join(history_dir, f"{MODEL_NAME.lower()}_val_{DATASET.lower()}_losses.npy"),
            arr=val_losses,
            allow_pickle=True
        )


        # Train Metrics
        # Acc
        val_metrics[epoch, 0] = val_acc
        # Recall
        val_metrics[epoch, 1] = val_recall
        # Precision
        val_metrics[epoch, 2] = val_precision
        # F1-Score
        val_metrics[epoch, 3] = val_f1
        # Save it to directory
        np.save(
            file=os.path.join(history_dir, f"{MODEL_NAME.lower()}_val_{DATASET.lower()}_metrics.npy"),
            arr=val_metrics,
            allow_pickle=True
        )

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            # print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{MODEL_NAME.lower()}_val_{DATASET.lower()}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Successfully saved at: {model_path}")



print("Finished.")
