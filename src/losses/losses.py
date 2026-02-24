# Import necessary libraries
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
from src.utils.config import device

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss calculation for 3D images.

    Args:
        penalty (str, optional): The type of penalty ('l1' or 'l2') to apply to the gradients. Default is 'l1'.
        loss_mult (float, optional): A multiplier for the computed gradient loss. Default is None.

    Methods:
        forward(y_pred, y_true): Computes the 3D gradient loss between predicted and ground truth images.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        """
        Computes the 3D gradient loss between predicted and ground truth images.

        Args:
            y_pred (Tensor): The predicted 3D image.
            y_true (Tensor): The ground truth 3D image.

        Returns:
            Tensor: The 3D gradient loss between the predicted and true images.
        """
        # Compute the differences in the y, x, and z gradients (for 3D images)
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        # Apply the penalty (L1 or L2) on the gradients
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        # Compute the total gradient loss by averaging over all three axes
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        # If loss multiplier is provided, apply it
        if self.loss_mult is not None:
            grad *= self.loss_mult
        
        return grad

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross-correlation (NCC) loss for image similarity.

    This loss computes the NCC over a sliding window applied to the input and predicted volumes.
    The loss is designed for use in image registration tasks where the goal is to match the features
    between two images.

    Attributes:
        win (list, optional): Size of the sliding window for NCC computation.
    """

    def __init__(self, win=None):
        """
        Initializes the NCC loss with an optional window size.

        Args:
            win (list, optional): Window size for NCC computation. Defaults to None (using a 9x9x9 window).
        """
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        """
        Computes the NCC loss between the true and predicted images.

        Args:
            y_true (torch.Tensor): Ground truth image.
            y_pred (torch.Tensor): Predicted image.

        Returns:
            torch.Tensor: The negative NCC loss value.
        """
        Ii = y_true  # Ground truth image
        Ji = y_pred  # Predicted image

        # Get the dimensionality of the input volume
        ndims = len(Ii[0].size()) - 1
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # Set window size if not provided
        win = [9] * ndims if self.win is None else self.win

        # Define filter for convolution
        sum_filt = torch.ones([1, 1, *win]).to(device)

        pad_no = math.floor(win[0] / 2)

        # Set stride and padding based on dimensionality
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Choose convolution function (e.g., for 3D images)
        conv_fn = getattr(F, 'conv3d')

        # Compute squared and product terms for NCC
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # Compute the NCC cross-correlation and variance
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute NCC and return negative value as loss
        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
