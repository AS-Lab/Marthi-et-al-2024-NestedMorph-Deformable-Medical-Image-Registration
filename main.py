import argparse
import torch
from src.utils.config import device
from scripts.train import train_model  # Import the train_model function from train.py
from scripts.train_cyclemorph import train_model as train_cyclemorph  # Import the train_model function from train_cyclemorph.py

def parse_args():
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train Model for Image Registration')
    
    # Required arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory for T1 moving images')
    parser.add_argument('--dwi_dir', type=str, required=True, help='Directory for diffusion fixed images')
    parser.add_argument('--model_label', type=str, required=True, 
                        help='Model label (MIDIR, NestedMorph, NestedMorph_Lite, TransMorph, ViTVNet, VoxelMorph, CycleMorph)')
    
    # Optional arguments with default values
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--img_size', type=str, default="64,64,64", 
                        help='Image size (HxWxD) as a comma-separated string')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--cont_training', action='store_false', 
                        help='Continue training from a saved checkpoint')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    # Parse command-line arguments
    args = parse_args()
    
    # Automatically select the training function based on model_label
    if args.model_label.lower() == "cyclemorph":
        train_function = train_cyclemorph
    else:
        train_function = train_model

    # Call the selected train_model function with individual arguments
        train_function(
        t1_dir=args.t1_dir,
        dwi_dir=args.dwi_dir,
        model_label=args.model_label,
        epochs=args.epochs,
        img_size=args.img_size,
        lr=args.lr,
        batch_size=args.batch_size,
        cont_training=args.cont_training,
        device=device  # Pass it here
    )

# python main.py --t1_dir /path/to/t1 --dwi_dir /path/to/dwi --epochs 100 --img_size 64,64,64 --lr 2e-4 --batch_size 2 --cont_training --model_label NestedMorph

