# Import necessary libraries
import os
import glob
import torch
from torch.utils.data import Dataset
from .data_utils import pkload  # Import the pkload function from data_utils module
import matplotlib.pyplot as plt
from scipy import ndimage  # Import scipy's ndimage for image resizing
import numpy as np
import SimpleITK as sitk  # Import SimpleITK for image registration

class RegistrationDataset(Dataset):
    """
    A custom PyTorch Dataset class for image registration tasks.
    
    This dataset loads pairs of images (moving and fixed) and applies transformations,
    resizing, and affine registration to prepare them for training.
    """
    def __init__(self, data_path, atlas_path, transforms, img_size=(64, 64, 64)):
        """
        Initialize the RegistrationDataset.
        
        Args:
            data_path (list): List of file paths for the moving images.
            atlas_path (list): List of file paths for the fixed (atlas) images.
            transforms (callable): A function/transform to apply to the images.
            img_size (tuple): Desired output size of the images (default: (64, 64, 64)).
        """
        self.paths = data_path  # Store the paths to the moving images
        self.atlas_path = atlas_path  # Store the paths to the fixed (atlas) images
        self.transforms = transforms  # Store the transformation function
        self.img_size = img_size  # Store the desired image size

    def one_hot(self, img, C):
        """
        Convert a label image to one-hot encoding.
        
        Args:
            img (np.ndarray): The input label image.
            C (int): The number of classes (channels in the output).
            
        Returns:
            np.ndarray: The one-hot encoded image.
        """
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))  # Initialize output array
        for i in range(C):
            out[i,...] = img == i  # Set the i-th channel to 1 where the label is i
        return out

    def __getitem__(self, index):
        """
        Retrieve a pair of moving and fixed images, apply transformations, and return them.
        
        Args:
            index (int): Index of the dataset item to retrieve.
            
        Returns:
            tuple: A tuple containing the transformed moving and fixed images.
        """
        path = self.paths[index]  # Get the path to the moving image
        path1 = self.atlas_path[index]  # Get the path to the fixed (atlas) image
        try:
            x = pkload(path)  # Load the moving image from the pickle file
            y = pkload(path1)  # Load the fixed image from the pickle file
        except Exception as e:
            # Handle errors during loading
            print(f"Error loading pickle file at path: {path}")
            print(f"Exception: {e}")
            return None, None  # Return None values if loading fails

        # Resize the images to the desired size
        x = resize_volume(x, self.img_size)  # Resize the moving image
        y = resize_volume(y, self.img_size)  # Resize the fixed image

        # Normalize the images to the range [0, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize the moving image
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize the fixed image

        # Convert numpy arrays to SimpleITK images
        x = sitk.GetImageFromArray(x)  # Convert moving image to SimpleITK format
        y = sitk.GetImageFromArray(y)  # Convert fixed image to SimpleITK format

        # Perform affine registration
        registration_method = sitk.ImageRegistrationMethod()  # Initialize registration method
        initial_transform = sitk.CenteredTransformInitializer(y, x, sitk.AffineTransform(3))  # Initialize transform
        registration_method.SetInitialTransform(initial_transform)  # Set the initial transform
        registration_method.SetMetricAsMeanSquares()  # Use mean squares as the metric
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)  # Set optimizer
        registration_method.Execute(y, x)  # Execute the registration

        # Apply the transform to the moving image
        x_registered = sitk.Resample(x, y, initial_transform, sitk.sitkLinear, 0.0)  # Resample the moving image

        # Convert the registered images back to numpy arrays
        x = sitk.GetArrayFromImage(x_registered)  # Convert moving image back to numpy
        y = sitk.GetArrayFromImage(y)  # Convert fixed image back to numpy

        # Add a new axis to the images for compatibility with transforms
        x, y = x[None, ...], y[None, ...]  # Add batch dimension
        x, y = self.transforms([x, y])  # Apply transformations

        # Ensure the arrays are contiguous in memory
        x = np.ascontiguousarray(x)  # Make moving image contiguous
        y = np.ascontiguousarray(y)  # Make fixed image contiguous

        # Convert numpy arrays to PyTorch tensors
        x, y = torch.from_numpy(x), torch.from_numpy(y)  # Convert to tensors
        return x, y  # Return the transformed images

    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.paths)  # Return the number of moving images

def resize_volume(img, shape):
    """
    Resize a 3D volume to the desired shape using interpolation.
    
    Args:
        img (np.ndarray): The input 3D volume.
        shape (tuple): The desired output shape (depth, width, height).
        
    Returns:
        np.ndarray: The resized 3D volume.
    """
    # Set the desired dimensions
    desired_depth = shape[0]  # Desired depth
    desired_width = shape[1]  # Desired width
    desired_height = shape[2]  # Desired height

    # Get the current dimensions of the input volume
    current_depth = img.shape[-1]  # Current depth
    current_width = img.shape[0]  # Current width
    current_height = img.shape[1]  # Current height

    # Compute the scaling factors for each dimension
    depth = current_depth / desired_depth  # Depth scaling factor
    width = current_width / desired_width  # Width scaling factor
    height = current_height / desired_height  # Height scaling factor

    depth_factor = 1 / depth  # Inverse depth scaling factor
    width_factor = 1 / width  # Inverse width scaling factor
    height_factor = 1 / height  # Inverse height scaling factor

    # Resize the volume using scipy's ndimage.zoom function
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)  # Linear interpolation
    return img  # Return the resized volume