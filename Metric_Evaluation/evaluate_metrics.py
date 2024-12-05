import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import utils
from data import datasets, trans
import argparse

import sys
# adding main to the system path
sys.path.insert(0, './NestedMorph') # CycleMorph, MIDIR, TransMorph, Vit-V-Net, VoxelMorph
from models import NestedMorph  # CycleMorph, MIDIR, TransMorph, Vit-V-Net, VoxelMorph

def load_model(model_path, img_size):
    model = NestedMorph(img_size)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    return model

def calculate_model_size(model):
    """Calculate the size of the model parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6  # Convert to millions

def save_visualization(def_image, def_field, save_path):
    # Create a figure with 2 subplots (1 2D mesh + 1 quiver)
    fig = plt.figure(figsize=(20, 10))
    
    # Plot the deformation field as a 3D surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    def_field_mean = np.mean(def_field, axis=0)
    
    x = np.arange(def_field_mean.shape[1])
    y = np.arange(def_field_mean.shape[0])
    x, y = np.meshgrid(x, y)
    
    ax1.plot_surface(x, y, def_field_mean, cmap='viridis')
    
    # Remove titles, labels, and ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_zticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('Deformation Value')
    # ax1.set_zlabel('')
    
    # Create a quiver plot for the deformation field
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(np.zeros_like(def_field_mean), cmap='gray')
    dx, dy = np.gradient(def_field_mean)
    
    # Normalize the vectors for better visualization
    magnitude = np.sqrt(dx**2 + dy**2)
    dx /= magnitude
    dy /= magnitude
    
    step = 2  # Step size for arrows
    ax2.quiver(x[::step, ::step], y[::step, ::step], dx[::step, ::step], dy[::step, ::step], color='w', scale=1, angles='xy', scale_units='xy')
    
    # Remove titles, labels, and ticks
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def evaluate_metrics(t1_image, dwi_image, model, reg_model, save_path, file_prefix):
    t1_image, dwi_image = t1_image.cuda(), dwi_image.cuda()
    x_in = torch.cat((t1_image, dwi_image), dim=1)
    
    with torch.no_grad():
        output = model(x_in)

    def_out = reg_model(t1_image.cuda().float(), output[1].cuda())

    def_out_np = def_out.cpu().numpy().squeeze()
    dwi_image_np = dwi_image.cpu().numpy().squeeze()
    t1_image_np = t1_image.cpu().numpy().squeeze()
    
    # Extract middle slice index
    mid_slice = def_out_np.shape[1] // 2

    deformation_field = output[1].cpu().numpy().squeeze()
    # deformation_field = np.mean(deformation_field, axis=0)

    # Save the visualization
    save_visualization(
        def_out_np[:, :, mid_slice],
        deformation_field[:, :, :, mid_slice],
        os.path.join(save_path, f'{file_prefix}_visualization.png')
    )

    # Continue with metric calculations
    ssim_value = utils.calculate_similarity(def_out_np, dwi_image_np)
    mse_value = utils.calculate_mse(def_out_np, dwi_image_np)
    mae_value = utils.calculate_mae(def_out_np, dwi_image_np)
    ncc_value = utils.calculate_ncc(def_out_np, dwi_image_np)
    mi_value = utils.calculate_mutual_information(def_out_np, dwi_image_np)
    hd95_value = utils.calculate_hd95(def_out_np, dwi_image_np)
    jac_det = utils.jacobian_determinant_vxm(output[1].cpu().numpy()[0, :, :, :, :])
    jacob_value = (np.sum(jac_det <= 0) / np.prod(dwi_image.detach().cpu().numpy()[0, 0, :, :, :].shape))

    return {
        'ssim': ssim_value,
        'mse': mse_value,
        'mae': mae_value,
        'ncc': ncc_value,
        'mi': mi_value,
        'hd95': hd95_value,
        'jacob_value': jacob_value
    }

def main(args):
    model_path = args.model_path
    t1_dir = args.t1_dir
    dwi_dir = args.dwi_dir
    save_path = args.save_path

    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_size = (args.img_size, args.img_size, args.img_size)
    batch_size = args.batch_size  # Set batch_size to process the desired number of image pairs

    model = load_model(model_path, img_size)

    print(f"Model: {model_path}, T1 Directory: {t1_dir}, DWI Directory: {dwi_dir}, Save Path: {save_path}")
    
    # Print model size
    model_size = calculate_model_size(model)
    print(f"Model size: {model_size:.2f} million parameters")
    
    reg_model = utils.register_model(img_size, 'nearest')  # Initialize your registration model
    reg_model.cuda()

    # Prepare data
    t1_files = glob.glob(t1_dir + '*.pkl')
    dwi_files = glob.glob(dwi_dir + '*.pkl')

    main_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_files}

    paired_files = [(t1_file, main_dict.get(os.path.basename(t1_file).split('_')[0])) 
                    for t1_file in t1_files 
                    if os.path.basename(t1_file).split('_')[0] in main_dict]

    # Create dataset and dataloader
    composed_transforms = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    eval_set = datasets.RegistrationDataset([pair[0] for pair in paired_files], 
                               [pair[1] for pair in paired_files], 
                               transforms=composed_transforms)
    
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    metrics = {'ssim': [], 'mse': [], 'mae': [], 'ncc': [], 'mi': [], 'hd95': [], 'jacob_value': []}
    results_list = []

    # Evaluate the first two pairs
    for idx, data in enumerate(eval_loader):
        if idx >= 25:  # Stop after evaluating the first two pairs
            break
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        file_prefix = f'pair_{idx+1}'
        results = evaluate_metrics(x, y, model, reg_model, save_path, file_prefix)
        
        # Save results for each pair in a CSV
        results_list.append({'file_prefix': file_prefix, **results})
        
        for key in metrics:
            metrics[key].append(results[key])

    # Save detailed results to CSV
    metrics_df = pd.DataFrame(results_list)
    metrics_df.to_csv(os.path.join(save_path, 'metrics_evaluation.csv'), index=False)

    # Calculate mean and standard deviation for each metric
    metrics_mean = {key: np.mean(metrics[key]) for key in metrics}
    metrics_std = {key: np.std(metrics[key]) for key in metrics}

    # Save summary statistics to CSV
    summary_df = pd.DataFrame({
        'Metric': list(metrics_mean.keys()),
        'Mean': list(metrics_mean.values()),
        'Std Dev': list(metrics_std.values())
    })
    summary_df.to_csv(os.path.join(save_path, 'metrics_summary.csv'), index=False)

    # Print the results
    print("Evaluation Results:")
    for key in metrics:
        print(f"{key.upper()}: Mean = {metrics_mean[key]:.4f}, Std Dev = {metrics_std[key]:.4f}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate Medical Image Registration Model")
    parser.add_argument('--model_path', type=str, default='final_model.pth.tar', help='Path to the trained model')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 images')
    parser.add_argument('--dwi_dir', type=str, required=True, help='Directory containing DWI images')
    parser.add_argument('--save_path', type=str, default='./evaluation_results/', help='Path to save evaluation results')
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images (assumed cubic)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')

    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args)

# python evaluate_metrics.py --model_path 'path/to/your/model.pth.tar' --t1_dir 'path/to/T1_dir/' --dwi_dir 'path/to/DWI_dir/' --save_path './path/to/save/' --img_size 128 --batch_size 1
