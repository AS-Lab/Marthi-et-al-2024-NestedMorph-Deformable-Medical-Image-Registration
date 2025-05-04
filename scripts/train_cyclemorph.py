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
from src.models.cyclemorph.cycleMorph_model import cycleMorph
from src.models.cyclemorph.cycleMorph_model import CONFIGS as CONFIGS
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
    Train the CycleMorph model for image registration.
    
    Args:
        t1_dir (str): Directory containing T1 moving images.
        dwi_dir (str): Directory containing DWI fixed images.
        model_label (str): Label of the model to train (e.g., "CycleMorph").
        epochs (int): Number of training epochs.
        img_size (str): Image size as a comma-separated string (e.g., "64,64,64").
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        cont_training (bool): Whether to continue training from a saved checkpoint.
    """
    # Define the save directory for experiments and logs
    save_dir = 'cyclemorph_1_ncc_{}_diffusion_{}/'
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)  # Create directory for saving experiments
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)  # Create directory for saving logs
    sys.stdout = Logger('logs/' + save_dir)  # Redirect stdout to the logger

    epoch_start = 0  # Starting epoch (0 if not continuing training)
    img_size = tuple(map(int, img_size.split(',')))  # Convert image size string to tuple

    '''
    Initialize spatial transformation function
    '''
    reg_model = register_model(img_size, 'nearest')  # Initialize registration model
    reg_model.cuda()  # Move registration model to GPU

    # Initialize CycleMorph model
    opt = CONFIGS['Cycle-Morph-v0']  # Load configuration for CycleMorph
    model = cycleMorph()  # Initialize CycleMorph model
    model.initialize(opt)  # Set up the model with the configuration

    # Initialize lists to store training and validation metrics
    train_losses = []  # List to store training losses
    val_dice_new_y = []  # List to store validation Dice scores for y
    val_dice_new_x = []  # List to store validation Dice scores for x

    # If continuing training, load the best model checkpoint
    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / epochs, 0.9), 8)  # Adjust learning rate
        best_model_path = model_dir + natsorted(os.listdir(model_dir))[0]  # Find the best model checkpoint
        best_model = torch.load(best_model_path)['state_dict']  # Load the model state
        model.netG_A.load_state_dict(best_model)  # Load the state into the generator
    else:
        updated_lr = lr  # Use the initial learning rate

    '''
    Initialize training data transformations
    '''
    train_composed = transforms.Compose([
        RandomFlip(0),  # Randomly flip images
        NumpyType((np.float32, np.float32)),  # Convert images to numpy arrays
    ])

    # Get all .pkl files in the T1 and DWI directories
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    best_mse = 0  # Track the best validation Dice score
    for epoch in range(epoch_start, epochs):
        '''
        Training phase
        '''
        loss_all = AverageMeter()  # Track average training loss
        loss_net_a = AverageMeter()  # Track average network loss
        idx = 0  # Batch index
        for data in train_loader:
            idx += 1
            data = [t.cuda() for t in data]  # Move data to GPU
            x = data[0]  # Moving image (T1)
            y = data[1]  # Fixed image (DWI)
            model.set_input(x, y)  # Set input data for the model
            loss_out, loss_reg, loss_net = model.optimize_parameters()  # Optimize model parameters
            loss_all.update(loss_out, y.numel())  # Update average loss
            loss_net_a.update(loss_net, y.numel())  # Update average network loss
            logger.info('Iter {} of {} loss {:.4f}, Reg: {:.6f}, loss net a: {:.6f}'.format(
                idx, len(train_loader), loss_out, loss_reg, loss_net))

        '''
        Validation phase
        '''
        eval_dsc_new_y = AverageMeter()  # Track Dice score for y
        eval_dsc_new_x = AverageMeter()  # Track Dice score for x
        with torch.no_grad():
            for data in val_loader:
                data = [t.cuda() for t in data]  # Move data to GPU
                x = data[0]  # Moving image (T1)
                y = data[1]  # Fixed image (DWI)
                model.set_input(x, y)  # Set input data for the model
                model.test()  # Run the model in test mode
                visuals = model.get_test_data()  # Get the output visuals
                flow = visuals['flow_A']  # Get the flow field
                def_out = reg_model(x.cuda().float(), flow.cuda())  # Apply the flow field to the moving image
                dsc_new_y = similarity(def_out, y, multichannel=True)  # Compute Dice score for y
                dsc_new_x = similarity(def_out, x, multichannel=True)  # Compute Dice score for x
                eval_dsc_new_y.update(dsc_new_y, x.size(0))  # Update average Dice score for y
                eval_dsc_new_x.update(dsc_new_x, x.size(0))  # Update average Dice score for x
        best_mse = max(eval_dsc_new_y.avg, best_mse)  # Update the best validation Dice score

        # Log validation results
        logger.info('Epoch {} loss {:.4f} DSC Deformed vs Fixed {:.4f} DSC Deformed vs Moving {:.4f}'.format(
            epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg))

        # Save the last def_out and y images from the 32nd slice
        if idx == len(train_loader):  # Check if it's the last iteration of the epoch
            last_def_out = def_out
            last_y = y
            last_x = x

        # Append the values to the lists
        train_losses.append(loss_all.avg)
        val_dice_new_y.append(eval_dsc_new_y.avg)
        val_dice_new_x.append(eval_dsc_new_x.avg)

        # Save training stats to CSV
        save_to_csv(epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg, 'logs/'+save_dir+'training_stats.csv')
        
        # Save the last def_out and y images from the 32nd slice for every 10th epoch
        if (epoch) % 10 == 0 and last_def_out is not None and last_y is not None:
            save_images(last_def_out, last_y, last_x, epoch)
        
        # Reset meters
        loss_all.reset()

    # Save the final model checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.netG_A.state_dict(),
        'best_mse': best_mse,
    }, save_dir='experiments/'+save_dir, filename='final_model_cyclemorph.pth.tar')

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
    
    # Select the 32nd slice
    def_out_slice = def_out_np[0, 0, :, :, 31]
    y_slice = y_np[0, 0, :, :, 31]
    x_slice = x_np[0, 0, :, :, 31]
    
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
