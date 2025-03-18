import random
import pickle
import numpy as np
import torch

# Define a large constant for seeding
M = 2 ** 32 - 1  # Maximum value for a 32-bit unsigned integer

def init_fn(worker):
    """
    Initialize the random seed for a worker in a multi-processing environment.
    
    Args:
        worker (int): The worker ID or index.
    """
    # Generate a random seed and adjust it based on the worker ID
    seed = torch.LongTensor(1).random_().item()  # Generate a random seed
    seed = (seed + worker) % M  # Adjust the seed based on the worker ID
    np.random.seed(seed)  # Set the seed for NumPy
    random.seed(seed)  # Set the seed for Python's random module

def add_mask(x, mask, dim=1):
    """
    Add a one-hot encoded mask to a tensor along a specified dimension.
    
    Args:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor to be added.
        dim (int): The dimension along which the mask will be added.
        
    Returns:
        torch.Tensor: The tensor with the mask added.
    """
    # Unsqueeze the mask to match the dimension of the input tensor
    mask = mask.unsqueeze(dim)
    # Calculate the new shape of the tensor after adding the mask
    shape = list(x.shape)
    shape[dim] += 21  # Increase the size of the specified dimension by 21
    # Create a new tensor filled with zeros
    new_x = x.new(*shape).zero_()
    # Scatter the mask values into the new tensor
    new_x = new_x.scatter_(dim, mask, 1.0)
    # Define a slice to insert the original tensor into the new tensor
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)  # Slice to insert the original tensor
    new_x[s] = x  # Insert the original tensor into the new tensor
    return new_x

def sample(x, size):
    """
    Randomly sample elements from a tensor.
    
    Args:
        x (np.ndarray or torch.Tensor): The input tensor or array.
        size (int): The number of elements to sample.
        
    Returns:
        torch.Tensor: A tensor containing the sampled elements.
    """
    # Randomly sample indices from the input tensor
    i = random.sample(range(x.shape[0]), size)  # Sample indices
    return torch.tensor(x[i], dtype=torch.int16)  # Convert to a tensor

def pkload(fname):
    """
    Load a Python object from a pickle file.
    
    Args:
        fname (str): The filename of the pickle file.
        
    Returns:
        object: The Python object loaded from the pickle file.
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)  # Load and return the object from the pickle file

# Define a constant shape for the coordinates
_shape = (240, 240, 155)

def get_all_coords(stride):
    """
    Generate a grid of coordinates based on a given stride.
    
    Args:
        stride (int): The stride (spacing) between coordinates.
        
    Returns:
        torch.Tensor: A tensor containing the grid coordinates.
    """
    # Generate a grid of coordinates using NumPy's meshgrid function
    return torch.tensor(
        np.stack([v.reshape(-1) for v in
                  np.meshgrid(
                      *[stride // 2 + np.arange(0, s, stride) for s in _shape],
                      indexing='ij')],
                 -1), dtype=torch.int16)  # Convert to a tensor

# Define a constant tensor with a single zero
_zero = torch.tensor([0])

def gen_feats():
    """
    Generate normalized feature coordinates for a 3D grid.
    
    Returns:
        np.ndarray: A 3D grid of normalized feature coordinates.
    """
    # Define the dimensions of the grid
    x, y, z = 240, 240, 155
    # Generate a 3D grid of coordinates using NumPy's meshgrid function
    feats = np.stack(
        np.meshgrid(
            np.arange(x), np.arange(y), np.arange(z),
            indexing='ij'), -1).astype('float32')  # Convert to float32
    # Normalize the coordinates
    shape = np.array([x, y, z])  # Define the shape of the grid
    feats -= shape / 2.0  # Center the coordinates around zero
    feats /= shape  # Normalize the coordinates to the range [-0.5, 0.5]
    return feats