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
import nibabel as nib  # Import nibabel for NIfTI file support


class RegistrationDataset(Dataset):
    """
    A custom PyTorch Dataset class for image registration tasks.
    NOW SUPPORTS BOTH .pkl AND .nii.gz FILE FORMATS!
    
    This dataset loads pairs of images (moving and fixed) and applies transformations,
    resizing, and affine registration to prepare them for training.
    """
    def __init__(self, data_path, atlas_path, transforms, img_size=(64, 64, 64)):
        """
        Initialize the RegistrationDataset.
        
        Args:
            data_path (list): List of file paths for the moving images (.pkl or .nii.gz).
            atlas_path (list): List of file paths for the fixed (atlas) images (.pkl or .nii.gz).
            transforms (callable): A function/transform to apply to the images.
            img_size (tuple): Desired output size of the images (default: (64, 64, 64)).
        """
        self.paths = data_path  # Store the paths to the moving images
        self.atlas_path = atlas_path  # Store the paths to the fixed (atlas) images
        self.transforms = transforms  # Store the transformation function
        self.img_size = img_size  # Store the desired image size

    def _load_file(self, file_path):
        """
        Load data from either .pkl or .nii.gz file automatically.
        
        Args:
            file_path (str): Path to the file (.pkl or .nii.gz).
            
        Returns:
            np.ndarray: The loaded 3D volume as a numpy array.
        """
        if file_path.endswith('.pkl'):
            # Load pickle file using the existing pkload function
            try:
                data = pkload(file_path)
                
                # Convert to numpy if it's a tensor
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
                
                return data.astype(np.float32)
            
            except Exception as e:
                print(f"Error loading pickle file: {file_path}")
                print(f"Exception: {e}")
                raise
        
        elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            # Load NIfTI file using nibabel
            try:
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                
                # Handle 4D volumes (take first volume if multi-volume)
                if data.ndim == 4:
                    print(f"Warning: 4D volume detected in {os.path.basename(file_path)}, using first volume")
                    data = data[..., 0]
                
                # Ensure 3D
                if data.ndim != 3:
                    raise ValueError(f"Expected 3D volume, got shape {data.shape}")
                
                return data.astype(np.float32)
            
            except Exception as e:
                print(f"Error loading NIfTI file: {file_path}")
                print(f"Exception: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use .pkl or .nii.gz")

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
        Automatically detects and loads .pkl or .nii.gz files.
        
        Args:
            index (int): Index of the dataset item to retrieve.
            
        Returns:
            tuple: A tuple containing the transformed moving and fixed images.
        """
        path = self.paths[index]  # Get the path to the moving image
        path1 = self.atlas_path[index]  # Get the path to the fixed (atlas) image
        
        try:
            # Load images (auto-detect format)
            x = self._load_file(path)  # Load the moving image
            y = self._load_file(path1)  # Load the fixed image
        except Exception as e:
            # Handle errors during loading
            print(f"Error loading files:")
            print(f"  Moving: {path}")
            print(f"  Fixed: {path1}")
            print(f"  Exception: {e}")
            return None, None  # Return None values if loading fails

        # Resize the images to the desired size
        x = resize_volume(x, self.img_size)  # Resize the moving image
        y = resize_volume(y, self.img_size)  # Resize the fixed image

        # Normalize the images to the range [0, 1]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Avoid division by zero
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)  # Normalize the moving image
        else:
            x = np.zeros_like(x)
            
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)  # Normalize the fixed image
        else:
            y = np.zeros_like(y)

        # Convert numpy arrays to SimpleITK images
        x_sitk = sitk.GetImageFromArray(x)  # Convert moving image to SimpleITK format
        y_sitk = sitk.GetImageFromArray(y)  # Convert fixed image to SimpleITK format

        # Perform affine registration
        registration_method = sitk.ImageRegistrationMethod()  # Initialize registration method
        initial_transform = sitk.CenteredTransformInitializer(
            y_sitk, x_sitk, sitk.AffineTransform(3)
        )  # Initialize transform
        registration_method.SetInitialTransform(initial_transform)  # Set the initial transform
        registration_method.SetMetricAsMeanSquares()  # Use mean squares as the metric
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=100
        )  # Set optimizer
        
        try:
            registration_method.Execute(y_sitk, x_sitk)  # Execute the registration
        except Exception as e:
            print(f"Warning: Registration failed for index {index}, using original images")
            print(f"Exception: {e}")
            # Continue without registration if it fails
            x_registered = x_sitk

        # Apply the transform to the moving image
        x_registered = sitk.Resample(
            x_sitk, y_sitk, initial_transform, sitk.sitkLinear, 0.0
        )  # Resample the moving image

        # Convert the registered images back to numpy arrays
        x = sitk.GetArrayFromImage(x_registered)  # Convert moving image back to numpy
        y = sitk.GetArrayFromImage(y_sitk)  # Convert fixed image back to numpy

        # Add a new axis to the images for compatibility with transforms
        x, y = x[None, ...], y[None, ...]  # Add batch dimension
        
        # Apply transformations if provided
        if self.transforms:
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
    img = ndimage.zoom(
        img, (width_factor, height_factor, depth_factor), order=1
    )  # Linear interpolation
    return img  # Return the resized volume


# Utility function to load files from directory (supporting both formats)
def get_file_list(directory, file_extension=None):
    """
    Get list of files from directory, supporting both .pkl and .nii.gz formats.
    
    Args:
        directory (str): Directory path
        file_extension (str, optional): Specific extension to filter ('pkl' or 'nii.gz').
                                       If None, returns all supported formats.
    
    Returns:
        list: Sorted list of file paths
    """
    if file_extension == 'pkl':
        files = glob.glob(os.path.join(directory, '*.pkl'))
    elif file_extension in ['nii.gz', 'nii']:
        files = glob.glob(os.path.join(directory, '*.nii.gz'))
        files += glob.glob(os.path.join(directory, '*.nii'))
    else:
        # Get both formats
        files = glob.glob(os.path.join(directory, '*.pkl'))
        files += glob.glob(os.path.join(directory, '*.nii.gz'))
        files += glob.glob(os.path.join(directory, '*.nii'))
    
    return sorted(files)
