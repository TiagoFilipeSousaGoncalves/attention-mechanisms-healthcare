# Imports
import numpy as np
import _pickle as cPickle
import os
from PIL import Image

# Sklearn Import
from sklearn.metrics import accuracy_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision


# Project Imports
from model_utilities import DenseNet121
from cbis_data_utilities import map_images_and_labels, TorchDatasetFromNumpyArray


# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBIS_proprocessed"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Results and Weights
weights_dir = os.path.join("results", "weights")
if os.path.isdir(weights_dir) == False:
    os.makedirs(weights_dir)


# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Choose Model Name
MODEL_NAME = 'DenseNet121'


# Data
# X Dimensions
CHANNELS = 1
HEIGHT = 224
WIDTH = 224

# Y dimensions
_, _, NR_CLASSES = map_images_and_labels(dir=train_dir)


# Model
if MODEL_NAME == 'DenseNet121':
    # Mean and STD to Normalize
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Model instance
    model = DenseNet121(
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        nr_classes=NR_CLASSES
    )


# Hyper-parameters
EPOCHS = 300
LOSS = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = 8


# Load data
# Train
# Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Dataset
train_set = TorchDatasetFromNumpyArray(base_data_path=train_dir, transform=train_transforms)

# Train Dataloader
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


# Validation
# Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation Dataset
val_set = TorchDatasetFromNumpyArray(base_data_path=val_dir, transform=val_transforms)

# Validation Dataloader
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Go through the number of Epochs
for epoch in range(EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print(f"Training Phase")
    
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
        loss = LOSS(logits, labels)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()
        
        # Update batch losses
        run_train_loss += (loss.item() * images.size(0))

        # Concatenate lists
        y_train_true += list(labels.cpu().detach().numpy())
        
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        y_train_pred = torch.nn.Softmax()(logits)
        y_train_pred = torch.argmax(y_train_pred, dim=1)
        y_train_pred += list(y_train_pred.cpu().detach().numpy())
    

    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)

    # Compute Training Accuracy
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)

    # Print Statistics
    print(f"Train loss: {avg_train_loss} \tTrain accuracy: {train_acc}")

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss





    # Validation Loop
    print(f"Validation Phase")


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
            loss = LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_val_true += list(labels.cpu().detach().numpy())
            
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            y_val_pred = torch.nn.Softmax()(logits)
            y_val_pred = torch.argmax(y_val_pred, dim=1)
            y_val_pred += list(y_val_pred.cpu().detach().numpy())
        

        # Compute Average Train Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Training Accuracy
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)

        # Print Statistics
        print(f"Validation loss: {avg_val_loss} \tValidation accuracy: {val_acc}")

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving model...")

            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(weights_dir, "densenet121_baseline_cbis.pt"))

            print("Successfully saved.")


# Finish statement
print("Finished.")