# Import necessary libraries
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian window.

    Args:
        window_size (int): The size of the window (odd number).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A normalized Gaussian window of the specified size.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window used for image smoothing in SSIM calculation.

    Args:
        window_size (int): The size of the window.
        channel (int): The number of channels (depth).

    Returns:
        Variable: A 2D Gaussian window.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    """
    Creates a 3D Gaussian window used for image smoothing in 3D SSIM calculation.

    Args:
        window_size (int): The size of the window.
        channel (int): The number of channels (depth).

    Returns:
        Variable: A 3D Gaussian window.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) for 2D images.

    Args:
        img1 (Tensor): The first image tensor.
        img2 (Tensor): The second image tensor.
        window (Tensor): The Gaussian window for filtering.
        window_size (int): The window size for the Gaussian filter.
        channel (int): The number of channels in the images.
        size_average (bool): Whether to return the average SSIM score or not.

    Returns:
        Tensor: The SSIM score.
    """
    # Compute the mean of the images
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Square of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute the variance of the images
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants to avoid division by zero
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()  # Return the mean SSIM score
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # Return SSIM score per channel

def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) for 3D images.

    Args:
        img1 (Tensor): The first 3D image tensor.
        img2 (Tensor): The second 3D image tensor.
        window (Tensor): The Gaussian window for filtering.
        window_size (int): The window size for the Gaussian filter.
        channel (int): The number of channels in the images.
        size_average (bool): Whether to return the average SSIM score or not.

    Returns:
        Tensor: The SSIM score.
    """
    # Compute the mean of the images in 3D
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    # Square of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute the variance of the images in 3D
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants to avoid division by zero
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula in 3D
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()  # Return the mean SSIM score
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # Return SSIM score per channel


class SSIM(torch.nn.Module):
    """
    A PyTorch module to compute the 2D Structural Similarity Index (SSIM).

    Args:
        window_size (int): The size of the Gaussian window.
        size_average (bool): Whether to return the average SSIM score.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Forward pass to compute SSIM between two images.

        Args:
            img1 (Tensor): The first image tensor.
            img2 (Tensor): The second image tensor.

        Returns:
            Tensor: The SSIM score.
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    """
    A PyTorch module to compute the 3D Structural Similarity Index (SSIM).

    Args:
        window_size (int): The size of the Gaussian window.
        size_average (bool): Whether to return the average SSIM score.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Forward pass to compute SSIM between two 3D images.

        Args:
            img1 (Tensor): The first 3D image tensor.
            img2 (Tensor): The second 3D image tensor.

        Returns:
            Tensor: The SSIM score (1 - SSIM for loss).
        """
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (Tensor): The first image tensor.
        img2 (Tensor): The second image tensor.
        window_size (int, optional): The size of the Gaussian window used in SSIM calculation. Default is 11.
        size_average (bool, optional): Whether to average the SSIM across all pixels. Default is True.

    Returns:
        Tensor: The SSIM score between the two images.
    """
    # Extract the number of channels in the image
    (_, channel, _, _) = img1.size()
    # Create the SSIM window based on the window size and the number of channels
    window = create_window(window_size, channel)

    # If images are on GPU, move the window to the same device
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    
    window = window.type_as(img1)

    # Call the underlying SSIM computation function
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size=11, size_average=True):
    """
    Computes the 3D Structural Similarity Index (SSIM) between two 3D images.

    Args:
        img1 (Tensor): The first 3D image tensor.
        img2 (Tensor): The second 3D image tensor.
        window_size (int, optional): The size of the Gaussian window used in SSIM calculation. Default is 11.
        size_average (bool, optional): Whether to average the SSIM across all voxels. Default is True.

    Returns:
        Tensor: The 3D SSIM score between the two images.
    """
    # Extract the number of channels in the 3D image
    (_, channel, _, _, _) = img1.size()
    # Create the 3D SSIM window based on the window size and the number of channels
    window = create_window_3D(window_size, channel)

    # If images are on GPU, move the window to the same device
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    
    window = window.type_as(img1)

    # Call the underlying 3D SSIM computation function
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss calculation for image gradients.

    Args:
        penalty (str, optional): The type of penalty ('l1' or 'l2') to apply to the gradients. Default is 'l1'.
        loss_mult (float, optional): A multiplier for the computed gradient loss. Default is None.

    Methods:
        forward(y_pred, y_true): Computes the gradient loss between predicted and ground truth images.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        """
        Computes the gradient loss between predicted and ground truth images.

        Args:
            y_pred (Tensor): The predicted image.
            y_true (Tensor): The ground truth image.

        Returns:
            Tensor: The gradient loss between the predicted and true images.
        """
        # Compute the differences in the y (vertical) and x (horizontal) gradients
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        # Apply the penalty (L1 or L2) on the gradients
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        # Compute the total gradient loss by averaging over both axes
        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        # If loss multiplier is provided, apply it
        if self.loss_mult is not None:
            grad *= self.loss_mult
        
        return grad


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


class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss calculation with isotropic total variation for 3D images.

    This is a more advanced version of the gradient loss with isotropic total variation, focusing on spatial smoothness.

    Methods:
        forward(y_pred, y_true): Computes the gradient loss with isotropic total variation between predicted and ground truth images.
    """
    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1  # Placeholder for potential future parameters

    def forward(self, y_pred, y_true):
        """
        Computes the gradient loss with isotropic total variation between predicted and ground truth images.

        Args:
            y_pred (Tensor): The predicted 3D image.
            y_true (Tensor): The ground truth 3D image.

        Returns:
            Tensor: The gradient loss with isotropic total variation.
        """
        # Compute the differences in the y, x, and z gradients (for 3D images)
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])

        # Apply L2 penalty on the gradients
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        # Compute the isotropic total variation gradient loss
        d = torch.mean(torch.sqrt(dx + dy + dz + 1e-6))
        grad = d / 3.0
        
        return grad


class DisplacementRegularizer(torch.nn.Module):
    """
    Displacement Regularizer class that computes regularization energies (gradient-based or bending energy) for displacement fields.

    Attributes:
        energy_type (str): The type of regularizer to apply. It can be 'bending', 'gradient-l2', or 'gradient-l1'.
    """

    def __init__(self, energy_type):
        """
        Initializes the displacement regularizer with the chosen energy type.

        Args:
            energy_type (str): Type of energy regularizer ('bending', 'gradient-l2', or 'gradient-l1').
        """
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        """
        Computes the x-direction gradient of a given tensor.

        Args:
            fv (torch.Tensor): Input tensor for which gradients are computed.

        Returns:
            torch.Tensor: Gradient in the x-direction.
        """
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv):
        """
        Computes the y-direction gradient of a given tensor.

        Args:
            fv (torch.Tensor): Input tensor for which gradients are computed.

        Returns:
            torch.Tensor: Gradient in the y-direction.
        """
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv):
        """
        Computes the z-direction gradient of a given tensor.

        Args:
            fv (torch.Tensor): Input tensor for which gradients are computed.

        Returns:
            torch.Tensor: Gradient in the z-direction.
        """
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        """
        Computes the gradient of the displacement field in all three directions.

        Args:
            Txyz (torch.Tensor): Displacement tensor (with shape [batch_size, 3, ...]).
            fn (function): Gradient function (e.g., gradient_dx, gradient_dy, gradient_dz).

        Returns:
            torch.Tensor: Stack of gradients in x, y, and z directions.
        """
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        """
        Computes the gradient norm (either L1 or L2 norm) for the displacement field.

        Args:
            displacement (torch.Tensor): Displacement field.
            flag_l1 (bool, optional): If True, computes L1 norm. Defaults to False (L2 norm).

        Returns:
            torch.Tensor: Mean gradient norm.
        """
        # Calculate gradients in all directions
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        
        # Choose norm type (L1 or L2)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2

        return torch.mean(norms) / 3.0  # Mean of the norms

    def compute_bending_energy(self, displacement):
        """
        Computes the bending energy of the displacement field.

        Args:
            displacement (torch.Tensor): Displacement field.

        Returns:
            torch.Tensor: Bending energy.
        """
        # Compute gradients in all directions
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)

        # Compute second-order derivatives (curvatures)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)

        # Compute bending energy (sum of squared second derivatives and cross derivatives)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        """
        Computes the regularization energy based on the selected type.

        Args:
            disp (torch.Tensor): Displacement field.
            _ (optional): Placeholder for additional arguments.

        Returns:
            torch.Tensor: The computed energy based on the chosen type.
        """
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


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
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

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


class MIND_loss(torch.nn.Module):
    """
    MIND (Multiscale Image Normalization Descriptor) loss for image registration tasks.
    This loss is based on the self-similarity of image patches and the use of a descriptor 
    derived from the MIND-SSC (Self-Similarity and Contrast) approach, designed to
    capture structural similarity across different scales.

    Attributes:
        win (list, optional): The window size for calculating the descriptor. Default is None.
    """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win  # Window size for the descriptor calculation

    def pdist_squared(self, x):
        """
        Compute pairwise squared Euclidean distance matrix for a set of points.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_points, dimension).

        Returns:
            Tensor: A pairwise squared distance matrix.
        """
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0  # Set NaNs (from invalid distances) to 0
        dist = torch.clamp(dist, 0.0, np.inf)  # Clamp to remove negative distances
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        """
        Compute the MIND-SSC (Self-Similarity and Contrast) descriptor for an image. 
        The descriptor captures the similarity between patches of the image at different scales.

        Args:
            img (Tensor): Input image of shape (batch_size, channels, depth, height, width).
            radius (int, optional): The radius of the neighborhood for self-similarity. Default is 2.
            dilation (int, optional): The dilation factor for computing the descriptor. Default is 2.

        Returns:
            Tensor: MIND-SSC descriptor for the input image.
        """
        kernel_size = radius * 2 + 1  # Kernel size based on the radius

        # Define six-neighborhood pattern for self-similarity
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # Build comparison mask based on distance and neighborhood pattern
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # Shift indices for applying convolution
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # Compute patch-SSD (Sum of Squared Differences)
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation for normalization and contrast adjustment
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # Reorder tensor to match the original C++ implementation
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        """
        Compute the MIND loss between predicted and true images.

        Args:
            y_pred (Tensor): Predicted image tensor of shape (batch_size, channels, depth, height, width).
            y_true (Tensor): Ground truth image tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            Tensor: The computed MIND loss.
        """
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MutualInformation(torch.nn.Module):
    """
    Mutual Information (MI) loss for image registration tasks.
    The loss computes the mutual information between two images, which is a measure of statistical dependence
    and similarity between the two images.

    Attributes:
        sigma_ratio (float): The ratio of sigma to the bin width for Gaussian approximation. Default is 1.
        minval (float): The minimum value for the image intensity. Default is 0.
        maxval (float): The maximum value for the image intensity. Default is 1.
        num_bin (int): The number of bins used for computing the histogram. Default is 32.
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        # Create bin centers for the histogram
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        # Compute the sigma for Gaussian approximation
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        """
        Compute the mutual information (MI) between the true and predicted images.

        Args:
            y_true (Tensor): Ground truth image tensor.
            y_pred (Tensor): Predicted image tensor.

        Returns:
            Tensor: The computed mutual information.
        """
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total number of voxels

        # Reshape bin centers
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        # Compute image terms using Gaussian approximation
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # Compute joint and marginal distributions
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # Return the average MI across the batch

    def forward(self, y_true, y_pred):
        """
        Compute the negative mutual information (MI) loss.

        Args:
            y_true (Tensor): Ground truth image tensor.
            y_pred (Tensor): Predicted image tensor.

        Returns:
            Tensor: The negative MI loss.
        """
        return -self.mi(y_true, y_pred)


class localMutualInformation(torch.nn.Module):
    """
    This class computes the Local Mutual Information (MI) between two images (or volumes) based on non-overlapping patches.
    MI is a measure of the statistical dependence between two variables (in this case, images or volumes),
    calculated over patches. This implementation is adapted for 2D and 3D images using Gaussian approximation for MI estimation.

    Args:
        sigma_ratio (float): The ratio used to compute the standard deviation of the Gaussian approximation.
        minval (float): Minimum value of the image intensity (default is 0).
        maxval (float): Maximum value of the image intensity (default is 1).
        num_bin (int): Number of bins used for discretization (default is 32).
        patch_size (int): Size of the non-overlapping patches used to compute the MI (default is 5).
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        # Create bin centers for discretizing the image intensities.
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        # Sigma for the Gaussian approximation.
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        # Precompute constant term for Gaussian approximation.
        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        """
        Computes the local mutual information (MI) between two images or volumes, calculated over non-overlapping patches.

        Args:
            y_true (Tensor): Ground truth image/volume.
            y_pred (Tensor): Predicted image/volume.

        Returns:
            Tensor: The computed local MI value.
        """

        # Clip the predicted and true images to the [0, max_clip] range.
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        # Reshape bin centers for broadcasting.
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        # Determine the padding required to make the image dimensions divisible by patch size.
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))

        # Apply padding to the images.
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        # Reshape images into non-overlapping patches.
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        # Compute the Gaussian approximation of the MI for each patch.
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        # Compute the joint and marginal distributions.
        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        # Compute the final MI using the formula.
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)

        return mi.mean()

    def forward(self, y_true, y_pred):
        """
        The forward method computes the negative local MI between the predicted and true images.

        Args:
            y_true (Tensor): Ground truth image/volume.
            y_pred (Tensor): Predicted image/volume.

        Returns:
            Tensor: The negative local MI loss.
        """
        return -self.local_mi(y_true, y_pred)
