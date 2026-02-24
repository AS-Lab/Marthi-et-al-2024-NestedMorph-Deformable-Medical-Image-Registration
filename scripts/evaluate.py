"""
Evaluation Script for IXI and OASIS Datasets.

Computes registration quality metrics:
    - Dice Similarity Coefficient (DSC) per anatomical structure
    - Structural Similarity Index (SSIM)
    - 95th Percentile Hausdorff Distance (HD95)
    - Jacobian Determinant (% non-positive = topology violations)

Usage:
    # Evaluate on IXI
    python scripts/evaluate.py \
        --dataset ixi \
        --data_dir ./data/IXI \
        --atlas_path ./data/IXI/atlas.pkl \
        --model_label NestedMorph \
        --checkpoint experiments/IXI_NestedMorph_.../final_model_NestedMorph.pth.tar

    # Evaluate on OASIS
    python scripts/evaluate.py \
        --dataset oasis \
        --data_dir ./data/OASIS \
        --model_label NestedMorph \
        --checkpoint experiments/OASIS_NestedMorph_.../final_model_NestedMorph.pth.tar

    # Compare multiple models on IXI
    python scripts/evaluate.py \
        --dataset ixi \
        --data_dir ./data/IXI \
        --atlas_path ./data/IXI/atlas.pkl \
        --model_label NestedMorph VoxelMorph TransMorph \
        --checkpoint ckpt1.pth.tar ckpt2.pth.tar ckpt3.pth.tar
"""

import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import ndimage
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.utils import register_model
from src.utils.config import device


# ─────────────────────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────────────────────

def compute_dice_per_label(pred_seg, true_seg, labels=None):
    """
    Compute Dice Similarity Coefficient for each anatomical label.
    
    Args:
        pred_seg (np.ndarray): Predicted segmentation (H, W, D).
        true_seg (np.ndarray): Ground truth segmentation (H, W, D).
        labels (list): List of label IDs to evaluate. If None, auto-detect.
    
    Returns:
        dict: {label_id: dice_score}
    """
    if labels is None:
        labels = sorted(set(np.unique(true_seg).astype(int)) | set(np.unique(pred_seg).astype(int)))
        labels = [l for l in labels if l != 0]  # Skip background

    dice_scores = {}
    for label in labels:
        pred_mask = (pred_seg == label).astype(np.float64)
        true_mask = (true_seg == label).astype(np.float64)
        intersection = np.sum(pred_mask * true_mask)
        union = np.sum(pred_mask) + np.sum(true_mask)
        if union == 0:
            dice_scores[label] = np.nan  # Label not present in either
        else:
            dice_scores[label] = (2.0 * intersection) / (union + 1e-8)
    return dice_scores


def compute_ssim_3d(pred, true):
    """
    Compute SSIM for 3D volumes using slice-wise computation.
    
    Args:
        pred (np.ndarray): Predicted image (H, W, D).
        true (np.ndarray): Ground truth image (H, W, D).
    
    Returns:
        float: Mean SSIM across slices.
    """
    from skimage.metrics import structural_similarity as ssim
    data_range = true.max() - true.min()
    if data_range == 0:
        return 0.0
    scores = []
    for i in range(pred.shape[0]):
        s, _ = ssim(pred[i], true[i], full=True, data_range=data_range)
        scores.append(s)
    return float(np.mean(scores))


def compute_hd95(pred_seg, true_seg, label, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff Distance for a specific label.
    Uses distance transforms for efficiency.
    
    Args:
        pred_seg (np.ndarray): Predicted segmentation.
        true_seg (np.ndarray): Ground truth segmentation.
        label (int): Label to compute HD95 for.
        voxel_spacing (tuple): Voxel spacing in mm.
    
    Returns:
        float: HD95 in mm, or np.nan if label not present.
    """
    pred_mask = (pred_seg == label).astype(bool)
    true_mask = (true_seg == label).astype(bool)

    if not pred_mask.any() or not true_mask.any():
        return np.nan

    # Distance from pred surface to true surface
    dt_true = ndimage.distance_transform_edt(~true_mask, sampling=voxel_spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_mask, sampling=voxel_spacing)

    # Surface voxels (boundary)
    pred_border = pred_mask ^ ndimage.binary_erosion(pred_mask)
    true_border = true_mask ^ ndimage.binary_erosion(true_mask)

    if not pred_border.any() or not true_border.any():
        return np.nan

    # Distances from pred surface to nearest true, and vice versa
    dist_pred_to_true = dt_true[pred_border]
    dist_true_to_pred = dt_pred[true_border]

    all_distances = np.concatenate([dist_pred_to_true, dist_true_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_jacobian_det(flow):
    """
    Compute Jacobian determinant of the deformation field.
    
    Args:
        flow (np.ndarray): Flow field of shape (3, H, W, D).
    
    Returns:
        tuple: (percent_non_positive, mean_det, std_det)
    """
    # Convert flow to displacement field: identity + flow
    H, W, D = flow.shape[1:]
    
    # Compute spatial gradients of displacement field
    # flow shape: (3, H, W, D)
    # d(flow_x)/dx, d(flow_x)/dy, d(flow_x)/dz, etc.
    
    # Using central differences
    def gradient(f, axis):
        return np.gradient(f, axis=axis)
    
    # Jacobian matrix components
    # J = I + grad(flow), where I is identity
    J00 = 1.0 + gradient(flow[0], 0)  # d(x + flow_x)/dx
    J01 = gradient(flow[0], 1)         # d(x + flow_x)/dy
    J02 = gradient(flow[0], 2)         # d(x + flow_x)/dz
    J10 = gradient(flow[1], 0)         # d(y + flow_y)/dx
    J11 = 1.0 + gradient(flow[1], 1)  # d(y + flow_y)/dy
    J12 = gradient(flow[1], 2)         # d(y + flow_y)/dz
    J20 = gradient(flow[2], 0)         # d(z + flow_z)/dx
    J21 = gradient(flow[2], 1)         # d(z + flow_z)/dy
    J22 = 1.0 + gradient(flow[2], 2)  # d(z + flow_z)/dz
    
    # Determinant of 3x3 Jacobian
    det = (J00 * (J11 * J22 - J12 * J21) -
           J01 * (J10 * J22 - J12 * J20) +
           J02 * (J10 * J21 - J11 * J20))
    
    num_non_positive = np.sum(det <= 0)
    total = det.size
    pct_non_positive = 100.0 * num_non_positive / total
    
    return pct_non_positive, float(np.mean(det)), float(np.std(det))


# ─────────────────────────────────────────────────────────────
# Model Loading Helper
# ─────────────────────────────────────────────────────────────

def load_model(model_label, img_size, checkpoint_path):
    """Initialize and load a model from checkpoint. Same logic as train.py."""
    if model_label == "MIDIR":
        from src.models.midir.midir import CubicBSplineNet
        model = CubicBSplineNet(img_size)
    elif model_label == "NestedMorph":
        from src.models.nestedmorph import NestedMorph
        model = NestedMorph(img_size)
    elif model_label == "TransMorph":
        from src.models.transmorph.TransMorph import TransMorph, CONFIGS as CONFIGS_TM
        model = TransMorph(CONFIGS_TM['TransMorph'], img_size)
    elif model_label == "ViTVNet":
        from src.models.vitvnet.vitvnet import ViTVNet, CONFIGS as CONFIGS_ViT
        model = ViTVNet(CONFIGS_ViT['ViT-V-Net'], img_size)
    elif model_label == "VoxelMorph":
        from src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(img_size)
    else:
        raise ValueError(f"Unknown model: {model_label}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, img_size, dataset_name, model_label,
                   eval_labels=None, output_dir='results'):
    """
    Run evaluation on a test set and return metrics.
    
    Args:
        model: Loaded PyTorch model.
        test_loader: DataLoader returning (x, y, x_seg, y_seg, subject_id).
        img_size: Tuple of image dimensions.
        dataset_name: 'ixi' or 'oasis'.
        model_label: Name of the model.
        eval_labels: List of label IDs for DSC/HD95 evaluation.
        output_dir: Where to save results CSV.
    
    Returns:
        dict: Summary metrics.
    """
    stn_nearest = register_model(img_size, 'nearest')
    stn_nearest.to(device)

    all_dice = []
    all_ssim = []
    all_hd95 = []
    all_jac_pct = []
    all_jac_mean = []
    per_subject = []

    print(f"\nEvaluating {model_label} on {dataset_name.upper()} ({len(test_loader)} samples)...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y, x_seg, y_seg, subj_id = batch
            subj_id = subj_id[0] if isinstance(subj_id, (list, tuple)) else subj_id
            x, y = x.to(device), y.to(device)

            output = model((x, y))
            warped = output[0]
            flow = output[1]

            # Numpy conversions
            warped_np = warped.cpu().numpy()[0, 0]
            y_np = y.cpu().numpy()[0, 0]
            flow_np = flow.cpu().numpy()[0]  # (3, H, W, D)

            # SSIM
            ssim_val = compute_ssim_3d(warped_np, y_np)
            all_ssim.append(ssim_val)

            # Jacobian determinant
            jac_pct, jac_mean, jac_std = compute_jacobian_det(flow_np)
            all_jac_pct.append(jac_pct)
            all_jac_mean.append(jac_mean)

            subject_result = {
                'subject': subj_id,
                'ssim': ssim_val,
                'jac_pct_neg': jac_pct,
                'jac_mean': jac_mean,
            }

            # DSC and HD95 (only if segmentations are available)
            has_segs = x_seg.sum() > 0 and y_seg.sum() > 0
            if has_segs:
                warped_seg = stn_nearest(x_seg.to(device).float(), flow)
                warped_seg_np = warped_seg.cpu().numpy()[0, 0]
                y_seg_np = y_seg.cpu().numpy()[0, 0]

                # Round to nearest integer for label maps
                warped_seg_np = np.round(warped_seg_np).astype(int)
                y_seg_np = np.round(y_seg_np).astype(int)

                labels = eval_labels
                if labels is None:
                    labels = sorted(set(np.unique(y_seg_np).astype(int)))
                    labels = [l for l in labels if l != 0]

                dice_dict = compute_dice_per_label(warped_seg_np, y_seg_np, labels)
                mean_dice = np.nanmean(list(dice_dict.values()))
                all_dice.append(mean_dice)
                subject_result['mean_dsc'] = mean_dice

                # HD95 for each label
                hd95_vals = []
                for label in labels:
                    h = compute_hd95(warped_seg_np, y_seg_np, label)
                    if not np.isnan(h):
                        hd95_vals.append(h)
                if hd95_vals:
                    mean_hd95 = np.mean(hd95_vals)
                    all_hd95.append(mean_hd95)
                    subject_result['mean_hd95'] = mean_hd95

            per_subject.append(subject_result)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                status = f"  [{i+1}/{len(test_loader)}] SSIM={ssim_val:.4f}, |Jac|<0={jac_pct:.2f}%"
                if has_segs:
                    status += f", DSC={mean_dice:.4f}"
                print(status)

    # ─── Summary ───
    summary = {
        'model': model_label,
        'dataset': dataset_name,
        'n_samples': len(test_loader),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'jac_neg_pct_mean': np.mean(all_jac_pct),
        'jac_neg_pct_std': np.std(all_jac_pct),
    }
    if all_dice:
        summary['dsc_mean'] = np.mean(all_dice)
        summary['dsc_std'] = np.std(all_dice)
    if all_hd95:
        summary['hd95_mean'] = np.mean(all_hd95)
        summary['hd95_std'] = np.std(all_hd95)

    # Save per-subject results
    out_path = os.path.join(output_dir, f'{dataset_name}_{model_label}_eval.csv')
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        if per_subject:
            writer = csv.DictWriter(f, fieldnames=per_subject[0].keys())
            writer.writeheader()
            writer.writerows(per_subject)
    print(f"Per-subject results saved to: {out_path}")

    return summary


def print_comparison_table(summaries):
    """Print a comparison table of all evaluated models."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    # Header
    has_dsc = any('dsc_mean' in s for s in summaries)
    has_hd95 = any('hd95_mean' in s for s in summaries)

    header = f"{'Model':<20} {'SSIM':>12} {'|Jac|≤0 (%)':>14}"
    if has_dsc:
        header += f" {'DSC':>14}"
    if has_hd95:
        header += f" {'HD95 (mm)':>14}"
    print(header)
    print("-" * len(header))

    for s in summaries:
        row = f"{s['model']:<20} {s['ssim_mean']:.4f}±{s['ssim_std']:.4f}"
        row += f" {s['jac_neg_pct_mean']:>6.2f}±{s['jac_neg_pct_std']:.2f}"
        if has_dsc and 'dsc_mean' in s:
            row += f" {s['dsc_mean']:.4f}±{s['dsc_std']:.4f}"
        elif has_dsc:
            row += f" {'N/A':>14}"
        if has_hd95 and 'hd95_mean' in s:
            row += f" {s['hd95_mean']:.2f}±{s['hd95_std']:.2f}"
        elif has_hd95:
            row += f" {'N/A':>14}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate registration models on IXI/OASIS')
    parser.add_argument('--dataset', type=str, required=True, choices=['ixi', 'oasis'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root dataset directory')
    parser.add_argument('--atlas_path', type=str, default=None,
                        help='Path to atlas file (required for IXI)')
    parser.add_argument('--model_label', type=str, nargs='+', required=True,
                        help='Model name(s) to evaluate')
    parser.add_argument('--checkpoint', type=str, nargs='+', required=True,
                        help='Checkpoint path(s), one per model')
    parser.add_argument('--img_size', type=str, default="160,192,224")
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    if len(args.model_label) != len(args.checkpoint):
        raise ValueError("Number of --model_label and --checkpoint must match")

    if args.dataset == 'ixi' and args.atlas_path is None:
        raise ValueError("--atlas_path is required for IXI dataset")

    img_size = tuple(map(int, args.img_size.split(',')))
    test_dir = os.path.join(args.data_dir, 'Test')

    # Build test dataset
    if args.dataset == 'ixi':
        from src.data.ixi_dataset import IXIBrainInferDataset
        test_set = IXIBrainInferDataset(
            data_dir=test_dir, atlas_path=args.atlas_path, img_size=img_size
        )
    else:
        from src.data.oasis_dataset import OASISBrainInferDataset
        test_set = OASISBrainInferDataset(
            data_dir=test_dir, img_size=img_size, is_train=False
        )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # Evaluate each model
    summaries = []
    for model_label, ckpt_path in zip(args.model_label, args.checkpoint):
        model = load_model(model_label, img_size, ckpt_path)
        summary = evaluate_model(
            model, test_loader, img_size, args.dataset, model_label,
            output_dir=args.output_dir
        )
        summaries.append(summary)

    # Print comparison
    print_comparison_table(summaries)

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, f'{args.dataset}_comparison.csv')
    with open(summary_path, 'w', newline='') as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
