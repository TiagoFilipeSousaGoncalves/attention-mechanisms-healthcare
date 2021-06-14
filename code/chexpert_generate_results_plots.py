# Imports
import numpy as np
import matplotlib.pyplot as plt
import os


# Results directory
directory = os.path.join("results", "chexpert", "history")
figs_directory = os.path.join("results", "chexpert", "figures")


# Models
MODELS = ["densenet121", "resnet50", "vgg16"]

# Training Formats
TRAINING_MODE = ["baseline", "baselinedaug", "mldam", "mldamdaug"]



# Go through models
for model in MODELS:

    # Go through mode of training
    for train_mode in TRAINING_MODE:

        # Get filenames
        # Train
        train_losses = np.load(os.path.join(directory, f"{model}_{train_mode}_tr_losses.npy"), allow_pickle=True)
        train_metrics = np.load(os.path.join(directory, f"{model}_{train_mode}_tr_metrics.npy"), allow_pickle=True)
        
        # Validation
        val_losses = np.load(os.path.join(directory, f"{model}_{train_mode}_val_losses.npy"), allow_pickle=True)
        val_metrics = np.load(os.path.join(directory, f"{model}_{train_mode}_val_metrics.npy"), allow_pickle=True)



        # Plot losses
        plt.title(f"{model.upper()} | {train_mode.upper()}")
        plt.plot(range(len(train_losses)), train_losses, label="Train")
        plt.plot(range(len(val_losses)), val_losses, label="Validation")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(figs_directory, f"{model.lower()}_{train_mode.lower()}_loss.png"))
        # plt.show()
        plt.clf()


        # Plot metrics
        metrics_dict = {0:"Accuracy", 1:"Recall", 2:"Precision", 3:"F1-Score"}
        for metric in range(4):
            metric_name = metrics_dict[metric]

            plt.title(f"{model.upper()} | {train_mode.upper()}")
            plt.plot(range(train_metrics.shape[0]), train_metrics[:, metric], label="Train")
            plt.plot(range(val_metrics.shape[0]), val_metrics[:, metric], label="Validation")
            plt.ylabel(f"{metric_name}")
            plt.xlabel("Epoch")
            plt.legend()
            plt.savefig(os.path.join(figs_directory, f"{model.lower()}_{train_mode.lower()}_{metric_name.lower()}.png"))
            # plt.show()
            plt.clf()