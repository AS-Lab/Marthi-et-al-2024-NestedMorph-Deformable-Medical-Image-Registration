# Import necessary libraries
import os
import glob
import torch
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import SimpleITK as sitk

class RegistrationDataset(Dataset):
    """
    Custom Dataset for loading and performing affine registration of 3D medical images.

    Args:
        data_path (list): List of file paths for the images to be loaded.
        atlas_path (list): List of file paths for the atlas images.
        transforms (callable): A function/transform to apply to the images and labels.
    """
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        """
        Converts the given image to a one-hot encoded format.

        Args:
            img (ndarray): Input image.
            C (int): Number of classes.

        Returns:
            ndarray: One-hot encoded version of the input image.
        """
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        """
        Loads and processes the image and atlas pair from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the processed image and label tensors.
        """
        path = self.paths[index]
        path1 = self.atlas_path[index]
        
        try:
            x = pkload(path)  # Load image
            y = pkload(path1)  # Load atlas image
            y = np.mean(y, axis=-1)  # Take the mean across the last axis of the atlas
        except Exception as e:
            print(f"Error loading pickle file at path: {path}")
            print(f"Exception: {e}")
            return None, None  # Return None if an error occurs

        # Resize images to 128x128x128
        x = resize_volume(x, (128, 128, 128))
        y = resize_volume(y, (128, 128, 128))

        # Normalize images to range [0, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Convert numpy arrays to SimpleITK images
        x = sitk.GetImageFromArray(x)
        y = sitk.GetImageFromArray(y)

        # Perform affine registration using SimpleITK
        registration_method = sitk.ImageRegistrationMethod()
        initial_transform = sitk.CenteredTransformInitializer(y, x, sitk.AffineTransform(3))
        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.Execute(y, x)

        # Apply the registration transform to the image
        x_registered = sitk.Resample(x, y, initial_transform, sitk.sitkLinear, 0.0)

        # Convert the registered image back to numpy
        x = sitk.GetArrayFromImage(x_registered)
        y = sitk.GetArrayFromImage(y)

        # Add a batch dimension and apply transformations
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        # Ensure arrays are contiguous and convert to torch tensors
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        
        return x, y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.paths)

def resize_volume(img, shape):
    """
    Resizes the input 3D image to the desired shape using zooming across the z-axis.

    Args:
        img (ndarray): The input 3D image.
        shape (tuple): The target shape (depth, width, height).

    Returns:
        ndarray: The resized image.
    """
    # Set the desired depth, width, and height
    desired_depth = shape[0]
    desired_width = shape[1]
    desired_height = shape[2]
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]

    # Compute the scaling factors for each dimension
    depth_factor = current_depth / desired_depth
    width_factor = current_width / desired_width
    height_factor = current_height / desired_height

    # Resize the image using scipy's zoom function
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    return img
