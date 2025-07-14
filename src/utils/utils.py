# Import necessary libraries
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
# import pystrum.pynd.ndutils as nd
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import re

class AverageMeter(object):
    """
    Computes and stores the average and current value.

    This class is useful for tracking the running average of values over multiple iterations,
    typically used in training loops to monitor metrics like loss or accuracy.
    """
    def __init__(self):
        self.reset()  # Initialize the attributes upon object creation

    def reset(self):
        """
        Resets the stored values.
        """
        self.val = 0  # Current value of the metric
        self.avg = 0  # Average value of the metric
        self.sum = 0  # Sum of all values encountered
        self.count = 0  # Number of values encountered
        self.vals = []  # List to store individual values for standard deviation calculation
        self.std = 0  # Standard deviation of the values

    def update(self, val, n=1):
        """
        Updates the stored values with a new value and its count.

        Args:
            val (float): The new value to update.
            n (int): The count of how many times this value should contribute. Default is 1.
        """
        self.val = val  # Store the current value
        self.sum += val * n  # Update the sum by multiplying the value with the count (n)
        self.count += n  # Update the total count
        self.avg = self.sum / self.count  # Update the average
        self.vals.append(val)  # Append the current value to the list of values
        self.std = np.std(self.vals)  # Recalculate the standard deviation based on all values


def pad_image(img, target_size):
    """
    Pads an image to the specified target size.

    Args:
        img (Tensor): The image tensor to be padded.
        target_size (tuple): A tuple of target dimensions (height, width, depth).

    Returns:
        Tensor: The padded image tensor.
    """
    # Calculate the number of rows, columns, and slices to pad
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)

    # Pad the image with zeros (constant padding) to the target size
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer Module.

    This module performs spatial transformations such as warping or shifting an image or a tensor
    based on a flow field. The flow field indicates how each pixel or voxel should be displaced.
    It can be used for tasks such as image registration, alignment, or geometric transformations.

    Args:
        size (tuple): The shape of the input tensor (height, width, depth).
        mode (str): The interpolation mode to use ('bilinear' or 'nearest').
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode  # Set interpolation mode (bilinear or nearest)

        # Create the sampling grid based on the provided size
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)  # Create a meshgrid for N-D grid coordinates
        grid = torch.stack(grids)  # Stack the grid coordinates into a tensor
        grid = torch.unsqueeze(grid, 0)  # Add a batch dimension
        grid = grid.type(torch.FloatTensor).cuda()  # Move the grid to the GPU

        # Register the grid as a buffer to keep it in the model state without saving to state dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        """
        Performs the spatial transformation on the input tensor.

        Args:
            src (Tensor): The source tensor to transform.
            flow (Tensor): The flow field tensor that defines the displacement of each point in the source.

        Returns:
            Tensor: The transformed tensor after applying the flow field.
        """
        # Compute the new locations by adding the flow to the original grid
        new_locs = self.grid + flow
        shape = flow.shape[2:]  # Get the spatial shape (height, width, depth)

        # Normalize the grid values to [-1, 1] for the resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Rearrange the new locations and change the order of coordinates for N-D data
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)  # For 2D data, swap the last dimension
            new_locs = new_locs[..., [1, 0]]  # Reverse the channel order for 2D
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)  # For 3D data, swap the last dimension
            new_locs = new_locs[..., [2, 1, 0]]  # Reverse the channel order for 3D

        # Apply grid sampling to get the transformed image using the flow and interpolation mode
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class register_model(nn.Module):
    """
    A PyTorch model for performing spatial image registration.

    The model uses a SpatialTransformer to apply a flow field to an image, which can be used for tasks such as image alignment
    or deformable image registration.

    Args:
        img_size (tuple): Size of the input image (default is (64, 256, 256)).
        mode (str): Interpolation mode to use ('bilinear' or 'nearest', default is 'bilinear').
    """
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)  # Initialize the spatial transformer module

    def forward(self, x, y):
        """
        Forward pass of the model.

        Applies the spatial transformation to the input image using the given flow field.

        Args:
            x (Tensor): The input image tensor (should be on GPU).
            y (Tensor): The flow field tensor (should be on GPU).

        Returns:
            Tensor: The transformed image after applying the flow.
        """
        img = x.cuda()  # Move the input image to GPU
        flow = y.cuda()  # Move the flow field to GPU
        out = self.spatial_trans(img, flow)  # Apply spatial transformation
        return out  # Return the transformed image


def dice_val(y_pred, y_true, num_clus):
    """
    Computes the Dice Similarity Coefficient (DSC) between predicted and true labels.

    This metric is commonly used for evaluating segmentation tasks in medical imaging.

    Args:
        y_pred (Tensor): The predicted labels (usually obtained from a model).
        y_true (Tensor): The true ground truth labels.
        num_clus (int): The number of classes (clusters) in the segmentation.

    Returns:
        Tensor: The mean Dice Similarity Coefficient (DSC) across all samples.
    """
    # Convert the predictions and ground truth to one-hot encoding
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)  # Remove the channel dimension (axis 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()  # Permute to [batch, num_classes, ...]
    
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

    # Compute the intersection and union for each class
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])  # Sum across spatial dimensions
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])  # Sum across spatial dimensions

    # Compute the Dice coefficient (2 * intersection / (union + epsilon))
    dsc = (2. * intersection) / (union + 1e-5)  # Avoid division by zero with epsilon
    return torch.mean(torch.mean(dsc, dim=1))  # Return mean DSC over the batch

def write2csv(line, name):
    """
    Writes a line of data to a CSV file.

    Args:
        line (str): The data to be written to the CSV file.
        name (str): The name of the CSV file.
    """
    with open(name+'.csv', 'a') as file:
        file.write(line)  # Write the line to the file
        file.write('\n')   # Add a newline after the line

def dice(y_pred, y_true):
    """
    Computes the Dice similarity coefficient (DSC) between two binary masks.

    Args:
        y_pred (np.ndarray): Predicted binary mask.
        y_true (np.ndarray): True binary mask.

    Returns:
        float: The Dice similarity coefficient.
    """
    intersection = y_pred * y_true  # Compute intersection
    intersection = np.sum(intersection)  # Sum of intersection
    union = np.sum(y_pred) + np.sum(y_true)  # Sum of union
    dsc = (2. * intersection) / (union + 1e-5)  # Compute Dice similarity coefficient
    return dsc

from skimage.metrics import structural_similarity as ssim

def similarity(y_pred, y_true, multichannel=False):
    """
    Computes the Structural Similarity Index (SSIM) for 3D volumes or images.

    Args:
        y_pred (np.ndarray): Predicted volumes/images, shape (B, D, H, W) or (B, D, H, W, C).
        y_true (np.ndarray): Ground truth volumes/images, same shape as y_pred.
        multichannel (bool, optional): Whether images are multichannel. Default is False for grayscale.

    Returns:
        float: The average SSIM score across slices.
    """
    # Ensure numpy format
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if present
    if y_pred.ndim == 5:
        y_pred = y_pred[0, 0, :, :, :]  # First batch and channel if 5D
        y_true = y_true[0, 0, :, :, :]

    ssim_scores = []
    data_range = y_true.max() - y_true.min()  # Compute dynamic range

    if multichannel:
        for i in range(y_pred.shape[0]):
            score, _ = ssim(y_pred[i], y_true[i], full=True, multichannel=True, data_range=data_range)
            ssim_scores.append(score)
    else:
        for i in range(y_pred.shape[0]):
            score, _ = ssim(y_pred[i], y_true[i], full=True, data_range=data_range)  # Add data_range
            ssim_scores.append(score)
    
    return np.mean(ssim_scores)  # Return average SSIM
