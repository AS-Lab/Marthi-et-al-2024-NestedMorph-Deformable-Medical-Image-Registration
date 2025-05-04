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
    N-Dimensional Spatial Transformer for warping images or volumes using a displacement field.

    This module performs spatial transformation by applying a flow (displacement field) to an input
    image or volume. It dynamically constructs a sampling grid, adds the flow to compute new coordinates,
    normalizes them, and uses `grid_sample` for interpolation.

    Args:
        mode (str): Interpolation mode to be used in grid sampling.
                    Options: 'bilinear', 'nearest', or 'bicubic' (for 4D inputs only).
                    Default is 'bilinear'.

    Forward Args:
        src (torch.Tensor): The input tensor to be warped.
                            Shape must be (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.

        flow (torch.Tensor): The displacement field indicating where to sample from `src`.
                             Shape must be (B, 2, H, W) for 2D or (B, 3, D, H, W) for 3D.
                             The last dimension size must match the spatial dimensions of `src`.

    Returns:
        torch.Tensor: The warped image or volume, same shape as `src`.
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        # Get spatial dimensions and number of dimensions
        shape = flow.shape[2:]  # e.g., (D, H, W) or (H, W)
        dim = len(shape)
        device = flow.device
        dtype = flow.dtype

        # Create normalized coordinate grid
        coords = torch.meshgrid([torch.arange(0, s, device=device, dtype=dtype) for s in shape], indexing='ij')
        grid = torch.stack(coords)  # Shape: (N, ...)
        grid = grid.unsqueeze(0).expand(flow.shape[0], -1, *shape)  # Shape: (B, N, ...)

        # Apply displacement field
        new_locs = grid + flow

        # Normalize coordinates to [-1, 1] range
        for i in range(dim):
            new_locs[:, i] = 2.0 * (new_locs[:, i] / (shape[i] - 1)) - 1.0

        # Rearrange dimensions to match grid_sample format
        if dim == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)      # (B, H, W, 2)
            new_locs = new_locs[..., [1, 0]]             # Convert to (x, y)
        elif dim == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)   # (B, D, H, W, 3)
            new_locs = new_locs[..., [2, 1, 0]]          # Convert to (x, y, z)

        # Warp the source image using the flow field
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


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
        self.spatial_trans = SpatialTransformer()  # Initialize the spatial transformer module

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


def dice_val_VOI(y_pred, y_true):
    """
    Computes the Dice Similarity Coefficient (DSC) for specific regions of interest (VOI).
    
    The VOIs are defined by a list of label values. This function computes the DSC for each VOI and returns the mean value.
    
    Args:
        y_pred (Tensor): The predicted labels (usually obtained from a model).
        y_true (Tensor): The true ground truth labels.

    Returns:
        float: The mean Dice Similarity Coefficient (DSC) for the specified VOIs.
    """
    # List of VOI (regions of interest) labels
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]

    # Detach and convert the tensors to numpy arrays
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]  # Extract prediction for the first sample
    true = y_true.detach().cpu().numpy()[0, 0, ...]  # Extract ground truth for the first sample

    # Initialize an array to store DSC values for each VOI
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0

    # Compute DSC for each VOI label
    for i in VOI_lbls:
        pred_i = pred == i  # Binary mask for predicted label
        true_i = true == i  # Binary mask for true label
        intersection = pred_i * true_i  # Calculate intersection (common pixels)
        intersection = np.sum(intersection)  # Sum intersection
        union = np.sum(pred_i) + np.sum(true_i)  # Sum union
        dsc = (2. * intersection) / (union + 1e-5)  # Compute DSC
        DSCs[idx] = dsc  # Store DSC value for the current VOI
        idx += 1

    return np.mean(DSCs)  # Return mean DSC across all VOIs


def jacobian_determinant_vxm(disp):
    """
    Computes the Jacobian determinant of a displacement field.

    The Jacobian determinant is useful for evaluating the local volume change induced by the flow (displacement).
    It is commonly used in deformation field analysis to measure the amount of expansion or contraction at each point.

    Args:
        disp (ndarray): The displacement field (2D or 3D) with shape [*vol_shape, nb_dims], where vol_shape is the 
                        shape of the spatial volume and nb_dims is the number of dimensions (2 or 3).

    Returns:
        ndarray: The Jacobian determinant of the displacement field.
    """
    # Transpose the displacement field to ensure correct shape
    disp = disp.transpose(1, 2, 3, 0)  # Reorder dimensions to match (x, y, z, dim)
    volshape = disp.shape[:-1]  # Get the spatial shape (height, width, depth)
    nb_dims = len(volshape)  # Get the number of dimensions (2D or 3D)
    
    # Ensure the displacement field is either 2D or 3D
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # Compute the grid of points corresponding to the spatial volume
    grid_lst = nd.volsize2ndgrid(volshape)  # Function to create a grid for the given volume size
    grid = np.stack(grid_lst, len(volshape))  # Stack the grid along the spatial dimensions

    # Compute the gradient of the displacement field
    J = np.gradient(disp + grid)  # Compute spatial gradients

    # If the displacement field is 3D, compute the 3D Jacobian determinant
    if nb_dims == 3:
        dx = J[0]  # Gradient along the x-axis
        dy = J[1]  # Gradient along the y-axis
        dz = J[2]  # Gradient along the z-axis

        # Compute the components of the Jacobian determinant using the cross-product
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2  # Return the Jacobian determinant for 3D

    else:  # For 2D displacement field
        dfdx = J[0]  # Gradient along the x-axis
        dfdy = J[1]  # Gradient along the y-axis

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]  # Compute the 2D Jacobian determinant


def process_label():
    """
    Processes labeling information for FreeSurfer segmentation.

    The function reads a text file 'label_info.txt' containing label information,
    looks for specified segment labels, and creates a mapping of segment index to label name.
    
    Returns:
        dict (dict): A dictionary where keys are segment indices and values are the corresponding 
                     segment names based on a predefined segmentation table.
    """
    # Predefined segmentation table (list of segment labels of interest)
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                 63, 72, 77, 80, 85, 251, 252, 253, 254, 255]

    # Read the label info from the file
    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []

    # Process each line in the label file
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])  # Try to convert the first element to an integer
            except:
                continue  # If it fails, skip the line
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])  # Store segment information
                dict[seg_i] = line[1]  # Create a dictionary mapping index to segment name
        seg_i += 1
    return dict


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


def dice_val_substruct(y_pred, y_true, std_idx):
    """
    Computes the Dice similarity coefficient (DSC) for each substructure label 
    between predicted and true segmentation masks.

    Args:
        y_pred (tensor): Predicted segmentation mask.
        y_true (tensor): True segmentation mask.
        std_idx (int): Index representing the substructure being processed.

    Returns:
        str: A string containing the DSC values for each substructure label.
    """
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)  # One-hot encode the predicted mask
        y_pred = torch.squeeze(y_pred, 1).permute(0, 4, 1, 2, 3).contiguous()  # Adjust dimensions
        y_true = nn.functional.one_hot(y_true, num_classes=46)  # One-hot encode the true mask
        y_true = torch.squeeze(y_true, 1).permute(0, 4, 1, 2, 3).contiguous()  # Adjust dimensions

    y_pred = y_pred.detach().cpu().numpy()  # Convert to numpy arrays
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)  # Prepare the output line starting with the index
    for i in range(46):
        pred_clus = y_pred[0, i, ...]  # Get the predicted binary mask for the current label
        true_clus = y_true[0, i, ...]  # Get the true binary mask for the current label
        intersection = pred_clus * true_clus
        intersection = intersection.sum()  # Compute intersection
        union = pred_clus.sum() + true_clus.sum()  # Compute union
        dsc = (2. * intersection) / (union + 1e-5)  # Compute Dice similarity coefficient
        line = line + ',' + str(dsc)  # Append the DSC value to the line
    return line


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


def smooth_seg(binary_img, sigma=1.5, thresh=0.4):
    """
    Applies Gaussian smoothing to a binary image and thresholds the result.

    Args:
        binary_img (np.ndarray): The binary image to be smoothed.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.5.
        thresh (float, optional): The threshold for binarizing the image after smoothing. Default is 0.4.

    Returns:
        np.ndarray: The smoothed and thresholded binary image.
    """
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)  # Apply Gaussian smoothing
    binary_img = binary_img > thresh  # Threshold the smoothed image
    return binary_img


def get_mc_preds(net, inputs, mc_iter: int = 25):
    """
    Performs Monte Carlo sampling for uncertainty estimation.

    Args:
        net (torch.nn.Module): The model (can be standard, MFVI, or MCDropout).
        inputs (torch.Tensor): The input to the model.
        mc_iter (int, optional): The number of Monte Carlo iterations. Default is 25.

    Returns:
        tuple: A tuple containing two lists - one for predicted images and one for flow predictions, 
               each containing the outputs from each MC iteration.
    """
    img_list = []
    flow_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)  # Forward pass through the network
            img_list.append(img)     # Store the predicted image
            flow_list.append(flow)   # Store the predicted flow
    return img_list, flow_list


def calc_uncert(tar, img_list):
    """
    Calculates the uncertainty by computing the squared differences between the predicted images 
    and the target image.

    Args:
        tar (torch.Tensor): The target (true) image.
        img_list (list): A list of predicted images from multiple Monte Carlo samples.

    Returns:
        torch.Tensor: The mean squared uncertainty value across the predicted images.
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2  # Compute squared difference for each prediction
        sqr_diffs.append(sqr_diff)  # Append to the list of squared differences
    # Concatenate all squared differences, compute the mean across the samples, and return the uncertainty
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def calc_error(tar, img_list):
    """
    Calculates the error by computing the squared differences between the predicted images 
    and the target image (similar to uncertainty).

    Args:
        tar (torch.Tensor): The target (true) image.
        img_list (list): A list of predicted images from multiple Monte Carlo samples.

    Returns:
        torch.Tensor: The mean squared error value across the predicted images.
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2  # Compute squared difference for each prediction
        sqr_diffs.append(sqr_diff)  # Append to the list of squared differences
    # Concatenate all squared differences, compute the mean across the samples, and return the error
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    """
    Performs Monte Carlo sampling to obtain predictions and calculates the error (MSE) for each sample.
    
    Args:
        net (torch.nn.Module): The model (can be standard, MFVI, or MCDropout).
        inputs (torch.Tensor): The input tensor for the model.
        target (torch.Tensor): The target (true) image.
        mc_iter (int, optional): The number of Monte Carlo iterations (samples). Default is 25.

    Returns:
        tuple: A tuple containing:
            - img_list (list): The list of predicted images from each Monte Carlo sample.
            - flow_list (list): The list of predicted flow fields from each Monte Carlo sample.
            - err (list): The list of MSE errors between the predicted and target images for each sample.
    """
    img_list = []
    flow_list = []
    MSE = nn.MSELoss()
    err = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)  # Forward pass to get prediction and flow
            img_list.append(img)     # Store the predicted image
            flow_list.append(flow)   # Store the predicted flow
            err.append(MSE(img, target).item())  # Compute and store the MSE error between prediction and target
    return img_list, flow_list, err


def get_diff_mc_preds(net, inputs, mc_iter: int = 25):
    """
    Performs Monte Carlo sampling to obtain predictions, flow, and displacement fields.

    Args:
        net (torch.nn.Module): The model (can be standard, MFVI, or MCDropout).
        inputs (torch.Tensor): The input tensor for the model.
        mc_iter (int, optional): The number of Monte Carlo iterations (samples). Default is 25.

    Returns:
        tuple: A tuple containing:
            - img_list (list): The list of predicted images from each Monte Carlo sample.
            - flow_list (list): The list of predicted flow fields from each Monte Carlo sample.
            - disp_list (list): The list of predicted displacement fields from each Monte Carlo sample.
    """
    img_list = []
    flow_list = []
    disp_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)  # Forward pass to get prediction, flow, and displacement
            img_list.append(img)     # Store the predicted image
            flow_list.append(flow)   # Store the predicted flow
            disp_list.append(disp)   # Store the predicted displacement
    return img_list, flow_list, disp_list


def uncert_regression_gal(img_list, reduction='mean'):
    """
    Computes uncertainty via Aleatoric and Epistemic components based on regression 
    of Monte Carlo samples.

    Args:
        img_list (list): A list of predicted uncertainty values (including aleatoric and epistemic).
        reduction (str, optional): The reduction method to apply to the results. 
            Options: 'mean' (default) or 'sum'.

    Returns:
        tuple: A tuple containing:
            - ale (torch.Tensor): The mean aleatoric uncertainty.
            - epi (torch.Tensor): The mean epistemic uncertainty.
            - uncert (torch.Tensor): The total uncertainty (sum of ale and epi).
    """
    img_list = torch.cat(img_list, dim=0)  # Concatenate all samples into a single tensor
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)  # Compute the mean over all predictions excluding the last component (assumed to be aleatoric)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)  # Compute the mean aleatoric uncertainty (last component)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)  # Compute epistemic uncertainty (variance over all predictions)

    # If there are multiple channels (e.g., RGB), average the variance over channels
    epi = epi.mean(dim=1, keepdim=True)  # Reduce to single channel for uncertainty
    uncert = ale + epi  # Total uncertainty is the sum of aleatoric and epistemic uncertainties

    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()  # Return raw uncertainties if no reduction specified

def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    """
    Computes the Uncertainty Consistency Loss (UCE Loss), which measures the consistency
    between the error and uncertainty in different bins of uncertainty.

    Args:
        errors (torch.Tensor): The prediction errors, usually the difference between predicted and true values.
        uncert (torch.Tensor): The uncertainty values, typically representing the uncertainty of the predictions.
        n_bins (int, optional): The number of bins to categorize uncertainty values. Default is 15.
        outlier (float, optional): The threshold for excluding bins with very few samples. Default is 0.0.
        range (tuple, optional): The range for binning the uncertainty values. If None, the bin boundaries are determined from `uncert`.

    Returns:
        tuple: A tuple containing:
            - uce (torch.Tensor): The total uncertainty consistency loss.
            - err_in_bin (torch.Tensor): The list of average errors in each bin.
            - avg_uncert_in_bin (torch.Tensor): The list of average uncertainties in each bin.
            - prop_in_bin (torch.Tensor): The proportion of data points in each bin.
    """
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)  # Initialize the UCE loss to 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Identify the points that fall within the current bin range
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # Calculate the proportion of points in the bin
        prop_in_bin_list.append(prop_in_bin)
        
        if prop_in_bin.item() > outlier:
            # Calculate the average error and uncertainty within the current bin
            errors_in_bin = errors[in_bin].float().mean()  # Mean error
            avg_uncert_in_bin = uncert[in_bin].mean()  # Mean uncertainty
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin  # Update the UCE loss

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin


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
