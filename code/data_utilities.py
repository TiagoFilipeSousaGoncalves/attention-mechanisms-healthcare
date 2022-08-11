# Imports
import os
import _pickle as cPickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
import torchvision
from torch.utils.data import Dataset



# Class: Torch Dataset from a NumPy array
class TorchDatasetFromNumpyArray(Dataset):
    def __init__(self, base_data_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.base_data_path = base_data_path
        imgs_labels, self.labels_dict, self.nr_classes = self.map_images_and_labels(directory=base_data_path)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)
    

    # Method: Get images and labels from directory files
    def map_images_and_labels(self, directory):
        
        # Images
        dir_files = os.listdir(directory)
        dir_imgs = [i for i in dir_files if i.split('.')[1]=='png']
        dir_imgs.sort()

        # Labels
        dir_labels_txt = [i.split('.')[0]+'case.txt' for i in dir_imgs]
        

        # Create a Numpy array to append file names and labels
        imgs_labels = np.zeros(shape=(len(dir_imgs), 2), dtype=object)

        # Go through images and labels
        idx = 0
        for image, label in zip(dir_imgs, dir_labels_txt):
            # Debug print
            # print(f"Image file: {image} | Label file: {label}")

            # Append image (Column 0)
            imgs_labels[idx, 0] = image
            
            # Append label (Column 1)
            # Read temp _label
            _label = np.genfromtxt(
                fname=os.path.join(dir, label),
                dtype=str
            )

            # Debug print
            # print(f"_label: {_label}")
            
            # Append to the Numpy Array
            imgs_labels[idx, 1] = str(_label)

            # Debug print
            # print(f"Image file: {imgs_labels[idx, 0]} | Label: {imgs_labels[idx, 1]}")


            # Update index
            idx += 1
        

        # Create labels dictionary to map strings into numbers
        _labels_unique = np.unique(imgs_labels[:, 1])

        # Nr of Classes
        nr_classes = len(_labels_unique)

        # Create labels dictionary
        labels_dict = dict()
        
        for idx, _label in enumerate(_labels_unique):
            labels_dict[_label] = idx


        return imgs_labels, labels_dict, nr_classes


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        
        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, img_name))
        
        # Perform transformations with Numpy Array
        image = np.asarray(image)
        image = np.reshape(image, newshape=(image.shape[0], image.shape[1], 1))
        image = np.concatenate((image, image, image), axis=2)

        # Load image with PIL
        image = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label



# Class: Torch Dataset from a picle file
class TorchDatasetFromPickle(Dataset):
    def __init__(self, base_data_path, pickle_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Init variables
        self.images_paths, self.images_labels = self.imgs_and_labels_from_pickle(base_data_path, pickle_path)
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)


    # Method: Get labels and paths from pickle
    def imgs_and_labels_from_pickle(self, base_data_path, pickle_path):
        # Open pickle file
        with open(pickle_path, "rb") as fp:
            pickle_data = cPickle.load(fp)

        # Split Images and Labels
        images_path = list()
        labels = list()

        # Go through pickle file
        for path, clf in zip(pickle_data[:, 0], pickle_data[:, 1]):
            images_path.append(os.path.join(base_data_path, path+".jpg"))
            labels.append(int(clf))


        # Assign variables to class variables
        images_paths = images_path
        images_labels = labels


        return images_paths, images_labels



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Get images
        img_name = self.images_paths[idx]
        image = Image.open(img_name)

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label



# Function: Get transforms
def get_transforms(split, data_aug, **kwargs):

    assert split in ("train", "validation", "test"), "Provide a valid data split: train, validation or test."

    # Train
    if split == "train":
        
        # Data Augmentation
        if data_aug:
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((kwargs["height"], kwargs["width"])),
            torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"])
            ])

        else:
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((kwargs["height"], kwargs["width"])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"])
            ])
    

    # Validation and Test
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((kwargs["height"], kwargs["width"])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"])
        ])


    return transforms



# Function: Unnormalize images
def unnormalize(image, mean_array, std_array):

    # Create a copy
    unnormalized_img = image.copy()

    # Get channels
    _, _, channels = unnormalized_img.shape


    for c in range(channels):
        unnormalized_img[:, :, c] = image[:, :, c] * std_array[c] + mean_array[c]


    return unnormalized_img



# Helper funtion to get figures to be shown after Captum VIZ
# https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure
def convert_figure(fig):

    # Create a dummy figure and use its manager to display "fig"  
    dummy = plt.figure(figsize=(6,6))
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
