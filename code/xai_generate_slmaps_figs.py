# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Captum Imports
from captum.attr import visualization as viz

# Project Imports
from data_utilities import convert_figure



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Choose the data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBIS", "MIMICCXR"], help="Choose the data set: CBIS, MIMICCXR.")


# Parse the arguments
args = parser.parse_args()


# Data set
DATASET = args.dataset



# Directories
results = os.path.join("results", {DATASET.lower()}, "attributes")
attribute_figs = os.path.join("results", {DATASET.lower()}, "attributes-figs")

# Models
models = ["densenet121", "resnet50", "vgg16"]


# Loop through the models
for model in models:
    
    # Get model directory
    model_dir = os.path.join(results, model)

    # Image save dir
    att_img_save_dir = os.path.join(attribute_figs, model)
    if not os.path.isdir(att_img_save_dir):
        os.makedirs(att_img_save_dir)


    # Get list of files
    attribute_flist = os.listdir(model_dir)
    attribute_flist = [i for i in attribute_flist if not i.startswith('.')]
    attribute_flist.sort()

    # Get images indices and labels
    imgs_indices = [i.split('_')[1] for i in attribute_flist]
    # print(imgs_indices)
    imgs_labels = [i.split('.')[0].split('_')[-1] for i in attribute_flist]
    # print(imgs_labels)


    # Loop through indices and labels
    for img_idx, img_label in zip(imgs_indices, imgs_labels):

        # Original Image
        original_fname = f"img_{img_idx}_original_{img_label}.npy"
        original_img = np.load(os.path.join(results, model, original_fname))
        figure, axis = viz.visualize_image_attr(None, original_img, method="original_image", title="Image", use_pyplot=False)
        convert_figure(figure)
        plt.savefig(os.path.join(att_img_save_dir, f"img_{img_idx}_original_{img_label}.png"))
        plt.clf()
        # plt.show()
        plt.close()


        # Baseline Saliency Map
        baseline_smap_fname = f"img_{img_idx}_dl_baseline_label_{img_label}.npy"
        baseline_smap = np.load(os.path.join(results, model, baseline_smap_fname))
        figure, axis = viz.visualize_image_attr(baseline_smap, original_img, method="blended_heat_map", sign="all", show_colorbar=True, title="Baseline", use_pyplot=False)
        convert_figure(figure)
        plt.savefig(os.path.join(att_img_save_dir, f"img_{img_idx}_dl_baseline_label_{img_label}.png"))
        plt.clf()
        # plt.show()
        plt.close()



        # Baseline Data Augmentation Saliency Map
        baselinedaug_smap_fname = f"img_{img_idx}_dl_baselinedaug_label_{img_label}.npy"
        baselinedaug_smap = np.load(os.path.join(results, model, baselinedaug_smap_fname))
        figure, axis = viz.visualize_image_attr(baselinedaug_smap, original_img, method="blended_heat_map", sign="all", show_colorbar=True, title="Baseline Data Aug.", use_pyplot=False)
        convert_figure(figure)
        plt.savefig(os.path.join(att_img_save_dir, f"img_{img_idx}_dl_baselinedaug_label_{img_label}.png"))
        plt.clf()
        # plt.show()
        plt.close()



        # MLDAM Saliency Map
        mldam_smap_fname = f"img_{img_idx}_dl_mldam_label_{img_label}.npy"
        mldam_smap = np.load(os.path.join(results, model, mldam_smap_fname))
        figure, axis = viz.visualize_image_attr(mldam_smap, original_img, method="blended_heat_map", sign="all", show_colorbar=True, title="MLDAM", use_pyplot=False)
        convert_figure(figure)
        plt.savefig(os.path.join(att_img_save_dir, f"img_{img_idx}_dl_mldam_label_{img_label}.png"))
        plt.clf()
        # plt.show()
        plt.close()



        # MLDAM Data Augmentation Saliency Map
        mldamdaug_smap_fname = f"img_{img_idx}_dl_mldamdaug_label_{img_label}.npy"
        mldamdaug_smap = np.load(os.path.join(results, model, mldamdaug_smap_fname))
        figure, axis = viz.visualize_image_attr(mldamdaug_smap, original_img, method="blended_heat_map", sign="all", show_colorbar=True, title="MLDAM Data Aug.", use_pyplot=False)
        convert_figure(figure)
        plt.savefig(os.path.join(att_img_save_dir, f"img_{img_idx}_dl_mldamdaug_label_{img_label}.png"))
        plt.clf()
        # plt.show()
        plt.close()



print("Finished")
