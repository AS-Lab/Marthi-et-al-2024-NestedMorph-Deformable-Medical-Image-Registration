"""
Inference Script for IXI Dataset.

Loads a trained model, runs on the IXI test set, and saves:
    - Warped images as .nii.gz
    - Deformation flow fields as .nii.gz
    - Warped segmentations as .nii.gz (if labels available)

Usage:
    python scripts/infer_ixi.py \
        --data_dir ./data/IXI \
        --atlas_path ./data/IXI/atlas.pkl \
        --model_label NestedMorph \
        --checkpoint experiments/IXI_NestedMorph_.../final_model_NestedMorph.pth.tar \
        --img_size 160,192,224
"""

import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.utils import register_model
from src.utils.config import device
from src.data.ixi_dataset import IXIBrainInferDataset


def save_nifti(data, path, affine=np.eye(4)):
    """Save tensor/array as .nii.gz"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    while data.ndim > 3:
        data = data[0]
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


def main():
    parser = argparse.ArgumentParser(description='Inference on IXI test set')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--atlas_path', type=str, required=True)
    parser.add_argument('--model_label', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', type=str, default="160,192,224")
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    img_size = tuple(map(int, args.img_size.split(',')))

    # Output directory
    out_dir = args.output_dir or f'results/IXI/{args.model_label}'
    os.makedirs(os.path.join(out_dir, 'warped'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'flow'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'warped_seg'), exist_ok=True)

    # Initialize model (same pattern as train.py)
    if args.model_label == "MIDIR":
        from src.models.midir.midir import CubicBSplineNet
        model = CubicBSplineNet(img_size)
    elif args.model_label == "NestedMorph":
        from src.models.nestedmorph import NestedMorph
        model = NestedMorph(img_size)
    elif args.model_label == "TransMorph":
        from src.models.transmorph.TransMorph import TransMorph, CONFIGS as CONFIGS_TM
        model = TransMorph(CONFIGS_TM['TransMorph'], img_size)
    elif args.model_label == "ViTVNet":
        from src.models.vitvnet.vitvnet import ViTVNet, CONFIGS as CONFIGS_ViT
        model = ViTVNet(CONFIGS_ViT['ViT-V-Net'], img_size)
    elif args.model_label == "VoxelMorph":
        from src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(img_size)
    else:
        raise ValueError(f"Unknown model: {args.model_label}")

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded {args.model_label} from {args.checkpoint}")

    # Spatial transformer for warping labels with nearest-neighbor
    stn_nearest = register_model(img_size, 'nearest')
    stn_nearest.to(device)

    # Build test dataset
    test_dir = os.path.join(args.data_dir, 'Test')
    test_set = IXIBrainInferDataset(
        data_dir=test_dir, atlas_path=args.atlas_path, img_size=img_size
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test subjects: {len(test_set)}")

    with torch.no_grad():
        for i, (x, y, x_seg, y_seg, subj_id) in enumerate(test_loader):
            subj_id = subj_id[0]
            x, y = x.to(device), y.to(device)
            output = model((x, y))
            warped = output[0]
            flow = output[1]

            # Save warped image
            save_nifti(warped, os.path.join(out_dir, 'warped', f'{subj_id}_warped.nii.gz'))

            # Save flow field (3, H, W, D) -> (H, W, D, 3) for NIfTI convention
            flow_np = flow.cpu().numpy()[0]  # (3, H, W, D)
            flow_nii = nib.Nifti1Image(flow_np.transpose(1, 2, 3, 0).astype(np.float32), np.eye(4))
            nib.save(flow_nii, os.path.join(out_dir, 'flow', f'{subj_id}_flow.nii.gz'))

            # Warp segmentation if available
            if x_seg.sum() > 0:
                warped_seg = stn_nearest(x_seg.to(device).float(), flow)
                save_nifti(warped_seg, os.path.join(out_dir, 'warped_seg', f'{subj_id}_warped_seg.nii.gz'))

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"  [{i+1}/{len(test_loader)}] {subj_id}")

    print(f"Done. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
