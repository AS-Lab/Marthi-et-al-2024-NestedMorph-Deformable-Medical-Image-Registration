"""
Data Format Conversion Utility.

Convert between .pkl (TransMorph format) and .nii.gz formats.

Usage:
    # Convert directory of .pkl to .nii.gz
    python scripts/convert_data.py \
        --input_dir data/IXI/Train \
        --output_dir data/IXI_nifti/Train \
        --to nii.gz

    # Convert directory of .nii.gz to .pkl
    python scripts/convert_data.py \
        --input_dir data/IXI_nifti/Train \
        --output_dir data/IXI_pkl/Train \
        --to pkl

    # Convert single file
    python scripts/convert_data.py \
        --input_file data/IXI/atlas.pkl \
        --output_dir data/IXI_nifti/ \
        --to nii.gz
"""

import os, sys, argparse, glob, pickle
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def pkl_to_nifti(pkl_path, output_dir, seg_suffix="_seg"):
    """
    Convert a .pkl file to .nii.gz.
    If the pkl contains (image, label) tuple, saves both.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    basename = os.path.basename(pkl_path).replace('.pkl', '')

    if isinstance(data, (tuple, list)):
        img = np.asarray(data[0], dtype=np.float32)
        nib.save(nib.Nifti1Image(img, np.eye(4)),
                 os.path.join(output_dir, f'{basename}.nii.gz'))
        print(f"  Saved: {basename}.nii.gz ({img.shape})")

        if len(data) >= 2 and data[1] is not None:
            seg = np.asarray(data[1], dtype=np.float32)
            nib.save(nib.Nifti1Image(seg, np.eye(4)),
                     os.path.join(output_dir, f'{basename}{seg_suffix}.nii.gz'))
            print(f"  Saved: {basename}{seg_suffix}.nii.gz ({seg.shape})")
    else:
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        img = np.asarray(data, dtype=np.float32)
        nib.save(nib.Nifti1Image(img, np.eye(4)),
                 os.path.join(output_dir, f'{basename}.nii.gz'))
        print(f"  Saved: {basename}.nii.gz ({img.shape})")


def nifti_to_pkl(nifti_path, output_dir, seg_suffix="_seg"):
    """
    Convert a .nii.gz file to .pkl.
    If a corresponding segmentation file exists, saves (image, label) tuple.
    """
    img = nib.load(nifti_path).get_fdata().astype(np.float32)
    if img.ndim == 4:
        img = img[..., 0]

    basename = os.path.basename(nifti_path)
    for ext in ['.nii.gz', '.nii']:
        if basename.endswith(ext):
            basename = basename.replace(ext, '')
            break

    # Check for segmentation
    seg = None
    for ext in ['.nii.gz', '.nii']:
        seg_path = nifti_path.replace(ext, f'{seg_suffix}{ext}')
        if os.path.exists(seg_path) and seg_path != nifti_path:
            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            if seg.ndim == 4:
                seg = seg[..., 0]
            break

    out_path = os.path.join(output_dir, f'{basename}.pkl')
    if seg is not None:
        with open(out_path, 'wb') as f:
            pickle.dump((img, seg), f)
        print(f"  Saved: {basename}.pkl (image + seg, shape {img.shape})")
    else:
        with open(out_path, 'wb') as f:
            pickle.dump(img, f)
        print(f"  Saved: {basename}.pkl (image only, shape {img.shape})")


def main():
    parser = argparse.ArgumentParser(description='Convert between .pkl and .nii.gz formats')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of files to convert')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Single file to convert')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--to', type=str, required=True, choices=['pkl', 'nii.gz'],
                        help='Target format')
    parser.add_argument('--seg_suffix', type=str, default='_seg',
                        help='Suffix for segmentation files (default: _seg)')
    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        raise ValueError("Must specify --input_dir or --input_file")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file:
        files = [args.input_file]
    else:
        if args.to == 'nii.gz':
            files = sorted(glob.glob(os.path.join(args.input_dir, '*.pkl')))
        else:
            files = sorted(glob.glob(os.path.join(args.input_dir, '*.nii.gz')))
            files += sorted(glob.glob(os.path.join(args.input_dir, '*.nii')))
            # Exclude segmentation files
            files = [f for f in files
                     if not f.replace('.nii.gz', '').replace('.nii', '').endswith(args.seg_suffix)]

    print(f"Converting {len(files)} files to .{args.to}")
    print(f"Output: {args.output_dir}\n")

    for fpath in files:
        try:
            if args.to == 'nii.gz':
                pkl_to_nifti(fpath, args.output_dir, args.seg_suffix)
            else:
                nifti_to_pkl(fpath, args.output_dir, args.seg_suffix)
        except Exception as e:
            print(f"  ERROR converting {os.path.basename(fpath)}: {e}")

    print(f"\nDone. {len(files)} files converted.")


if __name__ == '__main__':
    main()
