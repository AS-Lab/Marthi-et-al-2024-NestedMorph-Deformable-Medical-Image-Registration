import glob
from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader, random_split
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VoxelMorph
import csv
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # Number of parameters in millions

def main(args):
    batch_size = args.batch_size
    t1_dir = args.t1_dir
    dwi_dir = args.dwi_dir
    weights = [1, 1]  # loss weights
    save_dir = 'voxelmorph_1_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = args.lr
    epoch_start = 0
    max_epoch = args.epochs
    img_size = tuple(map(int, args.img_size.split(',')))
    cont_training = args.cont_training

    # Initialize lists to store the values
    train_losses = []
    val_dice_new_y = []
    val_dice_new_x = []

    '''
    Initialize model
    '''
    model = VoxelMorph(img_size)
    model.cuda()
    num_params = utils.count_parameters(model)
    print(f"Number of parameters in the model: {num_params:.2f} million parameters")

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([transforms.RandomFlip(0),
                                         transforms.NumpyType((np.float32, np.float32)),
                                         ])

    # Get all .pkl files in train and test directories
    t1_files = glob.glob(t1_dir + '*.pkl')
    dwi_files = glob.glob(dwi_dir + '*.pkl')

    # Create a dictionary to map the base filenames to the full paths
    main_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_files}

    # Pair train files with their corresponding test files
    paired_files = [(t1_file, main_dict.get(os.path.basename(t1_file).split('_')[0])) 
                    for t1_file in t1_files 
                    if os.path.basename(t1_file).split('_')[0] in main_dict]

    # Split the paired files into 80% training and 20% validation
    train_size = int(0.8 * len(paired_files))
    val_size = len(paired_files) - train_size

    train_paired_files, val_paired_files = random_split(paired_files, [train_size, val_size])

    # Create datasets
    train_set = datasets.RegistrationDataset([pair[0] for pair in train_paired_files], 
                                             [pair[1] for pair in train_paired_files], 
                                             transforms=train_composed)

    val_set = datasets.RegistrationDataset([pair[0] for pair in val_paired_files], 
                                           [pair[1] for pair in val_paired_files], 
                                           transforms=train_composed)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    
    for epoch in range(epoch_start, max_epoch):
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_in = torch.cat((x,y),dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        '''
        Validation
        '''
        eval_dsc_new_y = utils.AverageMeter()
        eval_dsc_new_x = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                def_out = reg_model(x.cuda().float(), output[1].cuda())

                dsc_new_y = utils.similarity(def_out, y, multichannel=True)
                dsc_new_x = utils.similarity(def_out, x, multichannel=True)
                eval_dsc_new_y.update(dsc_new_y, x.size(0))
                eval_dsc_new_x.update(dsc_new_x, x.size(0))
        best_dsc = max(eval_dsc_new_y.avg, best_dsc)

        logger.info('Epoch {} loss {:.4f} dice_new_y {:.4f} dice_new_x {:.4f}'.format(epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg))

        # Save the last def_out and y images from the 32nd slice
        if idx == len(train_loader):  # Check if it's the last iteration of the epoch
            last_def_out = def_out
            last_y = y
            last_x = x
        
        # Append the values to the lists
        train_losses.append(loss_all.avg)
        val_dice_new_y.append(eval_dsc_new_y.avg)
        val_dice_new_x.append(eval_dsc_new_x.avg)

        save_to_csv(epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg, 'logs/'+save_dir+'training_stats.csv')
        
        # Save the last def_out and y images from the 32nd slice for every 10th epoch
        if (epoch) % 10 == 0 and last_def_out is not None and last_y is not None:
            save_images(last_def_out, last_y, last_x, epoch)
        
        # Reset meters
        loss_all.reset()

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_dsc': best_dsc,
        'optimizer': optimizer.state_dict(),
    }, save_dir='experiments/' + save_dir, filename='final_model_voxelmorph.pth.tar')

    # Plot and save the curves
    plt.figure()
    plt.plot(range(epoch_start + 1, max_epoch + 1), train_losses, label='Training Loss')
    plt.plot(range(epoch_start + 1, max_epoch + 1), val_dice_new_y, label='Validation Dice Score (Y)')
    plt.plot(range(epoch_start + 1, max_epoch + 1), val_dice_new_x, label='Validation Dice Score (X)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('logs/' + save_dir + 'training_curves.png')

def save_to_csv(epoch, loss, dice_score_y, dice_score_x, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss, dice_score_y, dice_score_x])

def save_images(def_out, y, x, epoch):
    # Convert def_out, y, and x from tensors to numpy arrays
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
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU:', GPU_num)
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using:', torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available?', GPU_avai)

    # Argument parser
    parser = argparse.ArgumentParser(description='Train VoxelMorph Model for Image Registration')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory for T1 moving images')
    parser.add_argument('--dwi_dir', type=str, required=True, help='Directory for diffusion fixed images')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--img_size', type=str, default="64,64,64", help='Image size (HxWxD) as a comma-separated string')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--cont_training', action='store_true', help='Continue training from a saved checkpoint')

    args = parser.parse_args()
    main(args)

# python train_voxelmorph.py --t1_dir ./Registration_Data/T1_Moving/ --dwi_dir ./Registration_Data/Diffusion_Fixed/ --epochs 100 --img_size 128,128,128 --lr 2e-4 --batch_size 2 --cont_training

'''
Thank you to the authors of https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration 
for their work, which has greatly helped in the development of this project.
'''