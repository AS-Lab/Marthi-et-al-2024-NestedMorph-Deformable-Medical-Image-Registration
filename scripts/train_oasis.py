"""
Training Script for OASIS Dataset (Inter-Patient Brain MRI Registration).

Uses the same model initialization, losses, and training loop as the original train.py,
but with the OASIS inter-patient dataset instead of paired T1/DWI.

Usage:
    python scripts/train_oasis.py \
        --data_dir ./data/OASIS \
        --model_label NestedMorph \
        --epochs 500 \
        --img_size 160,192,224 \
        --lr 0.0001 \
        --batch_size 1
"""

import glob
from torch.utils.tensorboard import SummaryWriter
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.utils import *
from src.losses.losses import *
from src.data.data_utils import *
from src.data.trans import *
from src.data.rand import *
from src.data.oasis_dataset import OASISBrainDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


def save_to_csv(epoch, loss, ssim_y, ssim_x, csv_file):
    with open(csv_file, mode='a', newline='') as f:
        csv.writer(f).writerow([epoch, loss, ssim_y, ssim_x])


def save_images(def_out, y, x, epoch, save_dir):
    def_out_np = def_out.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()
    mid = def_out_np.shape[4] // 2

    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(def_out_np[0, 0, :, :, mid], cmap='gray'); axes[0].set_title('Deformed'); axes[0].axis('off')
    axes[1].imshow(y_np[0, 0, :, :, mid], cmap='gray'); axes[1].set_title('Fixed'); axes[1].axis('off')
    axes[2].imshow(x_np[0, 0, :, :, mid], cmap='gray'); axes[2].set_title('Moving'); axes[2].axis('off')
    plt.savefig(os.path.join(epoch_dir, f'mid_slice.png'), dpi=100)
    plt.close()


def train_model_oasis(data_dir, model_label, epochs, img_size, lr,
                       batch_size, cont_training, device):
    """
    Train a registration model on the OASIS dataset (inter-patient).
    Uses the same training loop structure as the original train.py.
    """
    weights = [1, 1]
    save_dir = f'OASIS_{model_label}_ncc_{weights[0]}_diffusion_{weights[1]}/'

    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    sys.stdout = Logger('logs/' + save_dir)

    epoch_start = 0
    img_size = tuple(map(int, img_size.split(',')))

    train_losses = []
    val_ssim_y = []
    val_ssim_x = []

    '''
    Initialize model (same as original train.py)
    '''
    if model_label == "MIDIR":
        from src.models.midir.midir import CubicBSplineNet
        model = CubicBSplineNet(img_size)
    elif model_label == "NestedMorph":
        from src.models.nestedmorph import NestedMorph
        model = NestedMorph(img_size)
    elif model_label == "TransMorph":
        from src.models.transmorph.TransMorph import TransMorph
        from src.models.transmorph.TransMorph import CONFIGS as CONFIGS_TM
        config = CONFIGS_TM['TransMorph']
        model = TransMorph(config, img_size)
    elif model_label == "ViTVNet":
        from src.models.vitvnet.vitvnet import ViTVNet
        from src.models.vitvnet.vitvnet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['ViT-V-Net']
        model = ViTVNet(config_vit, img_size)
    elif model_label == "VoxelMorph":
        from src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(img_size)
    else:
        raise ValueError(f"Unknown model label: {model_label}")

    model.to(device)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:.2f}M")

    reg_model = register_model(img_size, 'nearest')
    reg_model.to(device)

    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / epochs, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        model_dir = 'experiments/' + save_dir
        updated_lr = lr

    '''
    Build OASIS datasets
    '''
    train_composed = transforms.Compose([
        RandomFlip(0),
        NumpyType((np.float32, np.float32)),
    ])

    train_dir = os.path.join(data_dir, 'Train')
    val_dir = os.path.join(data_dir, 'Val')

    train_set = OASISBrainDataset(
        data_dir=train_dir,
        transforms=train_composed,
        img_size=img_size,
        is_train=True,
    )
    val_set = OASISBrainDataset(
        data_dir=val_dir,
        transforms=None,
        img_size=img_size,
        is_train=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    '''
    Training and Validation
    '''
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = NCC_vxm()
    criterions = [criterion, Grad3d(penalty='l2')]
    best_dsc = 0

    last_def_out = None  # FIX: initialise before epoch loop
    last_y = None
    last_x = None

    for epoch in range(epoch_start, epochs):
        '''
        Training phase
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, epochs, lr)

            x = data[0].to(device)
            y = data[1].to(device)

            output = model((x, y))

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[0], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()
            ))

        '''
        Validation phase
        '''
        eval_dsc_new_y = AverageMeter()
        eval_dsc_new_x = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                x = data[0].to(device)
                y = data[1].to(device)
                output = model((x, y))
                def_out = reg_model(x.float(), output[1])

                dsc_new_y = similarity(def_out, y, multichannel=True)
                dsc_new_x = similarity(def_out, x, multichannel=True)
                eval_dsc_new_y.update(dsc_new_y, x.size(0))
                eval_dsc_new_x.update(dsc_new_x, x.size(0))

        best_dsc = max(eval_dsc_new_y.avg, best_dsc)
        logger.info('Epoch {} loss {:.4f} SSIM(def,fix) {:.4f} SSIM(def,mov) {:.4f}'.format(
            epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg
        ))

        train_losses.append(loss_all.avg)
        val_ssim_y.append(eval_dsc_new_y.avg)
        val_ssim_x.append(eval_dsc_new_x.avg)

        save_to_csv(epoch, loss_all.avg, eval_dsc_new_y.avg, eval_dsc_new_x.avg,
                     'logs/' + save_dir + 'training_stats.csv')

        if idx == len(train_loader):
            last_def_out = def_out
            last_y = y
            last_x = x

        if epoch % 10 == 0 and last_def_out is not None:
            save_images(last_def_out, last_y, last_x, epoch, model_dir)

        loss_all.reset()

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_dsc': best_dsc,
        'optimizer': optimizer.state_dict(),
    }, save_dir='experiments/' + save_dir, filename=f'final_model_{model_label}.pth.tar')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_ssim_y, label='SSIM (Deformed vs Fixed)')
    plt.plot(range(1, epochs + 1), val_ssim_x, label='SSIM (Deformed vs Moving)')
    plt.xlabel('Epoch'); plt.ylabel('SSIM'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logs/' + save_dir + 'training_curves.png', dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train on OASIS Dataset (Inter-Patient)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root OASIS directory containing Train/, Val/, Test/')
    parser.add_argument('--model_label', type=str, required=True,
                        help='Model: MIDIR, NestedMorph, TransMorph, ViTVNet, VoxelMorph')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=str, default="160,192,224")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cont_training', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    from src.utils.config import device
    args = parse_args()
    train_model_oasis(
        data_dir=args.data_dir,
        model_label=args.model_label,
        epochs=args.epochs,
        img_size=args.img_size,
        lr=args.lr,
        batch_size=args.batch_size,
        cont_training=args.cont_training,
        device=device,
    )
