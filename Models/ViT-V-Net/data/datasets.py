import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
from scipy import ndimage 
import numpy as np


import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import SimpleITK as sitk


class RegistrationDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        path1 = self.atlas_path[index]
        try:
            x = pkload(path)
            y = pkload(path1)
            y = np.mean(y, axis=-1)
        except Exception as e:
            print(f"Error loading pickle file at path: {path}")
            print(f"Exception: {e}")
            # Handle the error, such as returning default values or raising an exception
            return None, None

        # Resize images to 64x64x64
        x = resize_volume(x, (128, 128, 128))
        y = resize_volume(y, (128, 128, 128))

        # Normalize x and y to the range [0, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Convert numpy arrays to SimpleITK images
        x = sitk.GetImageFromArray(x)
        y = sitk.GetImageFromArray(y)

        # Perform affine registration
        registration_method = sitk.ImageRegistrationMethod()
        initial_transform = sitk.CenteredTransformInitializer(y, x, sitk.AffineTransform(3))
        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.Execute(y, x)

        # Apply the transform to y
        x_registered = sitk.Resample(x, y, initial_transform, sitk.sitkLinear, 0.0)

        # Convert back to numpy arrays
        x = sitk.GetArrayFromImage(x_registered)
        y = sitk.GetArrayFromImage(y)

        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

def resize_volume(img, shape):
        """Resize across z-axis"""
        # Set the desired depth
        desired_depth = shape[0]
        desired_width = shape[1]
        desired_height = shape[2]
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]

        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height

        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height

        # Resize across z-axis
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

        return img