"""
IXI Dataset for Atlas-to-Patient Brain MRI Registration.

Compatible with TransMorph's preprocessed IXI dataset.
Supports both .pkl and .nii.gz file formats.

Expected directory structure:
    IXI_data/
    ├── Train/
    │   ├── subject_0.pkl        # Each .pkl = (image, label) tuple
    │   └── ...
    ├── Val/
    │   └── ...
    ├── Test/
    │   └── ...
    └── atlas.pkl                # Atlas image + label map

For .nii.gz format:
    ├── Train/
    │   ├── subject_0.nii.gz       # Brain MRI volume
    │   ├── subject_0_seg.nii.gz   # Segmentation labels (optional)
    │   └── ...
    └── atlas.nii.gz + atlas_seg.nii.gz

.pkl Format (TransMorph compatible):
    image, label = pickle.load(open("subject_0.pkl", "rb"))
    # image: shape (160, 192, 224), float32, intensity [0, 1]
    # label: shape (160, 192, 224), int, FreeSurfer subcortical labels
"""

import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
import nibabel as nib


def _discover_files(directory, label_suffix="_seg"):
    """
    Discover image files in a directory, supporting .pkl and .nii.gz.
    For .nii.gz, excludes segmentation files (those ending with label_suffix).
    """
    files = []
    # .pkl files
    files += sorted(glob.glob(os.path.join(directory, '*.pkl')))
    # .nii.gz files (exclude segmentation files)
    nii_files = sorted(glob.glob(os.path.join(directory, '*.nii.gz')))
    nii_files = [f for f in nii_files if not f.replace('.nii.gz', '').endswith(label_suffix)]
    files += nii_files
    # .nii files
    nii_files = sorted(glob.glob(os.path.join(directory, '*.nii')))
    nii_files = [f for f in nii_files if not f.replace('.nii', '').endswith(label_suffix)]
    files += nii_files
    return sorted(set(files))


def _load_volume_and_seg(file_path, label_suffix="_seg"):
    """
    Load an image volume and its corresponding segmentation from either .pkl or .nii.gz.
    
    For .pkl:  returns pickle.load() which should be (image, label) tuple.
    For .nii.gz: loads the image, then looks for <name>_seg.nii.gz as the label.
    
    Returns:
        (image, label) or (image, None) as np.float32 arrays.
    """
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, (tuple, list)):
            img = np.asarray(data[0], dtype=np.float32)
            seg = np.asarray(data[1], dtype=np.float32) if len(data) >= 2 else None
            return img, seg
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32), None
        elif isinstance(data, torch.Tensor):
            return data.numpy().astype(np.float32), None
        else:
            raise ValueError(f"Unexpected pkl contents in {file_path}: {type(data)}")

    elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
        nii = nib.load(file_path)
        img = nii.get_fdata().astype(np.float32)
        if img.ndim == 4:
            img = img[..., 0]
        # Try to find segmentation
        seg = None
        for ext in ['.nii.gz', '.nii']:
            if file_path.endswith(ext):
                seg_path = file_path.replace(ext, f'{label_suffix}{ext}')
                if os.path.exists(seg_path):
                    seg_nii = nib.load(seg_path)
                    seg = seg_nii.get_fdata().astype(np.float32)
                    if seg.ndim == 4:
                        seg = seg[..., 0]
                break
        return img, seg
    else:
        raise ValueError(f"Unsupported format: {file_path}. Use .pkl or .nii.gz")


def _normalize(img):
    """Normalize image to [0, 1]."""
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)


def _resize(img, target_shape, order=1):
    """Resize 3D volume to target shape."""
    if img.shape == tuple(target_shape):
        return img
    factors = [t / s for t, s in zip(target_shape, img.shape)]
    return ndimage.zoom(img, factors, order=order).astype(np.float32)


class IXIBrainDataset(Dataset):
    """
    IXI Brain MRI Dataset for atlas-to-patient registration.
    
    Returns 4 tensors: (moving, fixed, moving_seg, fixed_seg)
        - moving = atlas image (same for all samples)
        - fixed = patient image
        - moving_seg = atlas segmentation (same for all samples)
        - fixed_seg = patient segmentation
    
    All tensors have shape (1, H, W, D).
    Segmentations are zero tensors if labels are not available.
    """
    def __init__(self, data_dir, atlas_path, transforms=None,
                 img_size=(160, 192, 224), label_suffix="_seg"):
        """
        Args:
            data_dir (str): Directory containing subject files (Train/Val/Test).
            atlas_path (str): Path to atlas file (.pkl or .nii.gz).
            transforms: Optional transforms (e.g., RandomFlip from src.data.trans).
            img_size (tuple): Target image size. Use None to keep native resolution.
            label_suffix (str): Suffix for NIfTI segmentation files.
        """
        self.transforms = transforms
        self.img_size = tuple(img_size) if img_size else None
        self.label_suffix = label_suffix

        # Discover subject files
        self.file_paths = _discover_files(data_dir, label_suffix)
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .pkl or .nii.gz files found in {data_dir}")

        # Load atlas (shared across all samples)
        atlas_img, atlas_seg = _load_volume_and_seg(atlas_path, label_suffix)
        if self.img_size and atlas_img.shape != self.img_size:
            atlas_img = _resize(atlas_img, self.img_size, order=1)
            if atlas_seg is not None:
                atlas_seg = _resize(atlas_seg, self.img_size, order=0)
        self.atlas_img = _normalize(atlas_img)
        self.atlas_seg = atlas_seg

        print(f"[IXIBrainDataset] {len(self.file_paths)} subjects from {data_dir}")
        print(f"[IXIBrainDataset] Atlas: {atlas_path}, shape: {self.atlas_img.shape}")

    def __getitem__(self, index):
        # Load patient (fixed) image
        fpath = self.file_paths[index]
        y_img, y_seg = _load_volume_and_seg(fpath, self.label_suffix)

        # Resize if needed
        if self.img_size and y_img.shape != self.img_size:
            y_img = _resize(y_img, self.img_size, order=1)
            if y_seg is not None:
                y_seg = _resize(y_seg, self.img_size, order=0)

        y_img = _normalize(y_img)

        # Atlas = moving (x), Patient = fixed (y)
        x_img = self.atlas_img.copy()
        x_seg = self.atlas_seg.copy() if self.atlas_seg is not None else None

        # Add channel dimension -> (1, H, W, D)
        x_img = x_img[np.newaxis, ...]
        y_img = y_img[np.newaxis, ...]

        # Apply augmentation transforms
        if self.transforms:
            x_img, y_img = self.transforms([x_img, y_img])

        # Convert to tensors
        x = torch.from_numpy(np.ascontiguousarray(x_img)).float()
        y = torch.from_numpy(np.ascontiguousarray(y_img)).float()

        if x_seg is not None:
            x_seg_t = torch.from_numpy(np.ascontiguousarray(x_seg[np.newaxis, ...])).float()
        else:
            x_seg_t = torch.zeros_like(x)

        if y_seg is not None:
            y_seg_t = torch.from_numpy(np.ascontiguousarray(y_seg[np.newaxis, ...])).float()
        else:
            y_seg_t = torch.zeros_like(y)

        return x, y, x_seg_t, y_seg_t

    def __len__(self):
        return len(self.file_paths)


class IXIBrainInferDataset(IXIBrainDataset):
    """IXI dataset variant for inference. Returns subject ID as 5th element."""
    def __getitem__(self, index):
        x, y, x_seg, y_seg = super().__getitem__(index)
        subject_id = os.path.basename(self.file_paths[index]).split('.')[0]
        return x, y, x_seg, y_seg, subject_id
