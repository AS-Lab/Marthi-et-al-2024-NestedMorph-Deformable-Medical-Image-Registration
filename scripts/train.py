# Import necessary libraries
import glob
from torch.utils.tensorboard import SummaryWriter
import os, glob
from src.utils.utils import *
from src.losses.losses import *
from src.data.data_utils import *
from src.data.datasets import *
from src.data.trans import *
from src.data.rand import *
import sys
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Logger(object):
    """
    A custom logger class to redirect stdout to both the terminal and a log file.
    """
    def __init__(self, save_dir):
        """
        Initialize the logger.
        
        Args:
            save_dir (str): Directory where the log file will be saved.
        """
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(save_dir + "logfile.log", "a")  # Open a log file in append mode

    def write(self, message):
        """
        Write a message to both the terminal and the log file.
        
        Args:
            message (str): The message to log.
        """
        self.terminal.write(message)  # Write to terminal
        self.log.write(message)  # Write to log file

    def flush(self):
        """
        Flush the stream. This is required for compatibility with the logging module.
        """
        pass

def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.
        
    Returns:
        float: The number of parameters in millions.
    """
    return sum(p.numel() for p in model.parameters()) / 1e6  # Number of parameters in millions

def train_model(t1_dir, dwi_dir, model_label, epochs, img_size, lr, batch_size, cont_training):
    """
    Train the model for image registration.
    
    Args:
        t1_dir (str): Directory containing T1 moving images.
        dwi_dir (str): Directory containing DWI fixed images.
        model_label (str): Label of the model to train (e.g., "MIDIR", "NestedMorph", etc.).
        epochs (int): Number of training epochs.
        img_size (str): Image size as a comma-separated string (e.g., "64,64,64").
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        cont_training (bool): Whether to continue training from a saved checkpoint.
    """
    weights = [1, 1]  # Loss weights for the loss functions
    save_dir = f'{model_label}_1_ncc_{weights[0]}_diffusion_{weights[1]}/'  # Directory to save logs and experiments
    
    # Create directories for experiments and logs if they don't exist
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)  # Create directory for saving experiments
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)  # Create directory for saving logs
    
    # Redirect stdout to the logger
    sys.stdout = Logger('logs/' + save_dir)
    
    epoch_start = 0  # Starting epoch (0 if not continuing training)
    img_size = tuple(map(int, img_size.split(',')))  # Convert image size string to tuple

    # Initialize lists to store training and validation metrics
    train_losses = []  # List to store training losses
    val_dice_new_y = []  # List to store validation Dice scores for y
    val_dice_new_x = []  # List to store validation Dice scores for x

    '''
    Initialize model based on the model_label
    '''
    if model_label == "MIDIR":
        from src.models.midir.midir import CubicBSplineNet
        model = CubicBSplineNet(img_size)  # Initialize MIDIR model
    elif model_label == "NestedMorph":
        from src.models.nestedmorph import NestedMorph
        model = NestedMorph(img_size)  # Initialize NestedMorph model
    elif model_label == "TransMorph":
        from src.models.transmorph.TransMorph import TransMorph
        from src.models.transmorph.TransMorph import CONFIGS as CONFIGS_TM
        config = CONFIGS_TM['TransMorph']
        model = TransMorph(config, img_size)  # Initialize TransMorph model
    elif model_label == "ViTVNet":
        from src.models.vitvnet.vitvnet import ViTVNet
        from src.models.vitvnet.vitvnet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['ViT-V-Net']
        model = ViTVNet(config_vit, img_size)  # Initialize ViTVNet model
    elif model_label == "VoxelMorph":
        from src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(img_size)  # Initialize VoxelMorph model
    else:
        raise ValueError(f"Unknown model label: {model_label}")  # Raise error for unknown model

    model.cuda()  # Move the model to GPU
    num_params = count_parameters(model)  # Count the number of parameters in the model
    print(f"Number of parameters in the model: {num_params} million parameters")  # Print the number of parameters

    '''
    Initialize spatial transformation function
    '''
    reg_model = register_model(img_size, 'nearest')  # Initialize the registration model
    reg_model.cuda()  # Move the registration model to GPU

    '''
    If continuing training, load the best model checkpoint
    '''
    if cont_training:
        epoch_start = 0  # Reset epoch start if continuing training
        model_dir = 'experiments/' + save_dir  # Directory containing saved models
        updated_lr = round(lr * np.power(1 - (epoch_start) / epochs, 0.9), 8)  # Adjust learning rate
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']  # Load the best model checkpoint
        model.load_state_dict(best_model)  # Load the state into the model
    else:
        updated_lr = lr  # Use the initial learning rate

    '''
    Initialize training data transformations
    '''
    train_composed = transforms.Compose([
        RandomFlip(0),  # Randomly flip images
        NumpyType((np.float32, np.float32)),  # Convert images to numpy arrays
    ])

    # Load all .pkl files in the T1 and DWI directories
    t1_files = glob.glob(t1_dir + '*.pkl')  # List of T1 files
    dwi_files = glob.glob(dwi_dir + '*.pkl')  # List of DWI files

    # Create a dictionary to map base filenames to full paths
    main_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_files}

    # Pair T1 files with corresponding DWI files
    paired_files = [(t1_file, main_dict.get(os.path.basename(t1_file).split('_')[0]))
                    for t1_file in t1_files 
                    if os.path.basename(t1_file).split('_')[0] in main_dict]

    # Split the paired files into 80% training and 20% validation
    train_size = int(0.8 * len(paired_files))  # 80% for training
    val_size = len(paired_files) - train_size  # 20% for validation
    train_paired_files, val_paired_files = random_split(paired_files, [train_size, val_size])

    # Create datasets for training and validation
    train_set = RegistrationDataset(
        [pair[0] for pair in train_paired_files],  # List of T1 files for training
        [pair[1] for pair in train_paired_files],  # List of DWI files for training
        transforms=train_composed,  # Apply transformations
        img_size=img_size  # Set image size
    )

    val_set = RegistrationDataset(
        [pair[0] for pair in val_paired_files],  # List of T1 files for validation
        [pair[1] for pair in val_paired_files],  # List of DWI files for validation
        transforms=train_composed,  # Apply transformations
        img_size=img_size  # Set image size
    )

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    '''
    Training and Validation
    '''
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)  # Initialize optimizer
    criterion = NCC_vxm()  # NCC loss function
    criterions = [criterion, Grad3d(penalty='l2')]  # Add gradient loss
    best_dsc = 0  # Track the best validation Dice score

    for epoch in range(epoch_start, epochs):
        '''
        Training phase
        '''
        loss_all = AverageMeter()  # Track average training loss
        idx = 0  # Batch index
        for data in train_loader:
            idx += 1
            model.train()  # Set the model to training mode
            adjust_learning_rate(optimizer, epoch, epochs, lr)  # Adjust learning rate
            data = [t.cuda() for t in data]  # Move data to GPU
            x, y = data[0], data[1]  # Unpack moving and fixed images
            x_in = torch.cat((x, y), dim=1)  # Concatenate moving and fixed images
            output = model((x, y))  # Forward pass through the model
            
            # Compute loss
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[0], y) * weights[n]  # Compute individual losses
                loss_vals.append(curr_loss)
                loss += curr_loss  # Accumulate total loss
                
            loss_all.update(loss.item(), y.numel())  # Update average loss
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            # Log training progress
            logger.info('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()
            ))

        '''
        Validation phase
        '''
        eval_dsc_new_y = AverageMeter()  # Track Dice score for y
        eval_dsc_new_x = AverageMeter()  # Track Dice score for x
        with torch.no_grad():  # Disable gradient computation
            for data in val_loader:
                model.eval()  # Set the model to evaluation mode
                data = [t.cuda() for t in data]  # Move data to GPU
                x, y = data[0], data[1]  # Unpack moving and fixed images
                x_in = torch.cat((x, y), dim=1)  # Concatenate moving and fixed images
                output = model((x, y))  # Forward pass through the model
                def_out = reg_model(x.cuda().float(), output[1].cuda())  # Apply the flow field to the moving image

                # Compute Dice scores
                dsc_new_y = similarity(def_out, y, multichannel=True)  # Dice score for y
                dsc_new_x = similarity(def_out, x, multichannel=True)  # Dice score for x
                eval_dsc_new_y.update(dsc_new_y, x.size(0))  # Update average Dice score for y
                eval_dsc_new_x.update(dsc_new_x, x.size(0))  # Update average Dice score for x
                
        # Log validation results
        best_dsc = max(eval_dsc_new_y.avg, best_dsc)  # Update the best validation Dice score
        logger.info('Epoch {} loss {:.4f} DSC Deformed vs Fixed {:.4f} DSC Deformed vs Moving {:.4f}'.format(
            epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg
        ))

        # Append the values to the lists
        train_losses.append(loss_all.avg)
        val_dice_new_y.append(eval_dsc_new_y.avg)
        val_dice_new_x.append(eval_dsc_new_x.avg)

        # Save training stats to CSV
        save_to_csv(epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg, 'logs/'+save_dir+'training_stats.csv')
        
        # Save the last def_out and y images from the 32nd slice
        if idx == len(train_loader):  # Check if it's the last iteration of the epoch
            last_def_out = def_out
            last_y = y
            last_x = x
        
        # Save the last def_out and y images from the 32nd slice for every 10th epoch
        if (epoch) % 10 == 0 and last_def_out is not None and last_y is not None:
            save_images(last_def_out, last_y, last_x, epoch)
        
        # Reset meters
        loss_all.reset()

    # Save the final model checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_dsc': best_dsc,
        'optimizer': optimizer.state_dict(),
    }, save_dir='experiments/' + save_dir, filename=f'final_model_{model_label}.pth.tar')

    # Plot and save the training curves
    plt.figure()
    plt.plot(range(epoch_start + 1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(epoch_start + 1, epochs + 1), val_dice_new_y, label='Validation Dice Score (Y)')
    plt.plot(range(epoch_start + 1, epochs + 1), val_dice_new_x, label='Validation Dice Score (X)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('logs/' + save_dir + 'training_curves.png')

def save_to_csv(epoch, loss, dice_score_y, dice_score_x, csv_file):
    """
    Save training statistics to a CSV file.
    
    Args:
        epoch (int): Current epoch.
        loss (float): Training loss.
        dice_score_y (float): Validation Dice score for y.
        dice_score_x (float): Validation Dice score for x.
        csv_file (str): Path to the CSV file.
    """
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss, dice_score_y, dice_score_x])

def save_images(def_out, y, x, epoch):
    """
    Save the last def_out, y, and x images from the 32nd slice.
    
    Args:
        def_out (torch.Tensor): Deformed output image.
        y (torch.Tensor): Fixed image.
        x (torch.Tensor): Moving image.
        epoch (int): Current epoch.
    """
    # Convert tensors to numpy arrays
    def_out_np = def_out.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()
    
    # Compute the middle slice index dynamically
    middle_idx = def_out_np.shape[4] // 2  # Assuming the 5th dimension (index 4) is the depth

    # Select the middle slice
    def_out_slice = def_out_np[0, 0, :, :, middle_idx]
    y_slice = y_np[0, 0, :, :, middle_idx]
    x_slice = x_np[0, 0, :, :, middle_idx]
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot def_out image
    axes[0].imshow(def_out_slice, cmap='gray')
    axes[0].set_title('def_out (Epoch {})'.format(epoch))
    axes[0].axis('off')
    
    # Plot y image
    axes[1].imshow(y_slice, cmap='gray')
    axes[1].set_title('y (Epoch {})'.format(epoch))
    axes[1].axis('off')

    # Plot x image
    axes[2].imshow(x_slice, cmap='gray')
    axes[2].set_title('x (Epoch {})'.format(epoch))
    axes[2].axis('off')
    
    # Save the combined image
    plt.savefig('combined_epoch_{}.png'.format(epoch))
    plt.close()

def comput_fig(img):
    """
    Create a figure with multiple slices of an image.
    
    Args:
        img (torch.Tensor): Input image tensor.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    """
    Adjust the learning rate using a polynomial decay schedule.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        MAX_EPOCHES (int): Maximum number of epochs.
        INIT_LR (float): Initial learning rate.
        power (float): Power factor for decay.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    """
    Save a model checkpoint.
    
    Args:
        state (dict): Model state to save.
        save_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
        max_model_num (int): Maximum number of models to keep in the directory.
    """
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
