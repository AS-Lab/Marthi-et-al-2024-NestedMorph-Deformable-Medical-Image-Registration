"""
OASIS Dataset for Inter-Patient Brain MRI Registration.

Compatible with TransMorph's preprocessed OASIS dataset (Learn2Reg Task 03).
Supports both .pkl and .nii.gz file formats.

Expected directory structure:
    OASIS_data/
    ├── Train/          # 394 subjects
    │   ├── subject_0.pkl    # Each .pkl = (image, label) tuple
    │   └── ...
    ├── Val/            # 19 subjects
    │   └── ...
    └── Test/           # 38 subjects
        └── ...

Inter-patient registration:
    - Training: random pairs are formed each epoch
    - Val/Test: consecutive pairs (subject_0 -> subject_1, etc.)
"""

import os
import glob
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
import nibabel as nib

# Reuse helpers from ixi_dataset
from src.data.ixi_dataset import _discover_files, _load_volume_and_seg, _normalize, _resize


class OASISBrainDataset(Dataset):
    """
    OASIS Brain MRI Dataset for inter-patient registration.

    Returns 4 tensors: (moving, fixed, moving_seg, fixed_seg)
    All tensors have shape (1, H, W, D).
    
    During training: pairs are random (a different partner each time).
    During val/test: pairs are consecutive (idx 0->1, 2->3, ...).
    """
    def __init__(self, data_dir, transforms=None, img_size=(160, 192, 224),
                 label_suffix="_seg", is_train=True):
        """
        Args:
            data_dir (str): Directory containing subject files.
            transforms: Optional transforms.
            img_size (tuple): Target image size. None for native resolution.
            label_suffix (str): Suffix for NIfTI segmentation files.
            is_train (bool): If True, random pairing; otherwise consecutive pairing.
        """
        self.transforms = transforms
        self.img_size = tuple(img_size) if img_size else None
        self.label_suffix = label_suffix
        self.is_train = is_train

        # Discover subject files
        self.file_paths = _discover_files(data_dir, label_suffix)
        if len(self.file_paths) < 2:
            raise FileNotFoundError(
                f"Need >= 2 subjects for inter-patient registration, "
                f"found {len(self.file_paths)} in {data_dir}"
            )

        # Build pairs for val/test (consecutive subjects)
        if not is_train:
            self.pairs = []
            for i in range(0, len(self.file_paths) - 1, 2):
                self.pairs.append((i, i + 1))
            if len(self.file_paths) % 2 == 1:
                self.pairs.append((len(self.file_paths) - 1, len(self.file_paths) - 2))

        mode = "train (random pairs)" if is_train else f"eval ({len(self.pairs)} fixed pairs)"
        print(f"[OASISBrainDataset] {len(self.file_paths)} subjects from {data_dir} | {mode}")

    def _load_and_preprocess(self, file_path):
        """Load, resize, normalize a subject."""
        img, seg = _load_volume_and_seg(file_path, self.label_suffix)
        if self.img_size and img.shape != self.img_size:
            img = _resize(img, self.img_size, order=1)
            if seg is not None:
                seg = _resize(seg, self.img_size, order=0)
        img = _normalize(img)
        return img, seg

    def __getitem__(self, index):
        if self.is_train:
            idx_x = index
            idx_y = random.choice([i for i in range(len(self.file_paths)) if i != index])
        else:
            idx_x, idx_y = self.pairs[index]

        # Load moving (x) and fixed (y)
        x_img, x_seg = self._load_and_preprocess(self.file_paths[idx_x])
        y_img, y_seg = self._load_and_preprocess(self.file_paths[idx_y])

        # Add channel dimension
        x_img = x_img[np.newaxis, ...]
        y_img = y_img[np.newaxis, ...]

        # Augmentation
        if self.transforms:
            x_img, y_img = self.transforms([x_img, y_img])

        # Convert to tensors
        x = torch.from_numpy(np.ascontiguousarray(x_img)).float()
        y = torch.from_numpy(np.ascontiguousarray(y_img)).float()
        x_seg_t = torch.from_numpy(np.ascontiguousarray(x_seg[np.newaxis, ...])).float() \
            if x_seg is not None else torch.zeros_like(x)
        y_seg_t = torch.from_numpy(np.ascontiguousarray(y_seg[np.newaxis, ...])).float() \
            if y_seg is not None else torch.zeros_like(y)

        return x, y, x_seg_t, y_seg_t

    def __len__(self):
        if self.is_train:
            return len(self.file_paths)
        return len(self.pairs)


class OASISBrainInferDataset(OASISBrainDataset):
    """OASIS dataset variant for inference. Returns subject pair ID as 5th element."""
    def __getitem__(self, index):
        x, y, x_seg, y_seg = super().__getitem__(index)
        if self.is_train:
            pair_id = f"subj{index}"
        else:
            i, j = self.pairs[index]
            name_i = os.path.basename(self.file_paths[i]).split('.')[0]
            name_j = os.path.basename(self.file_paths[j]).split('.')[0]
            pair_id = f"{name_i}_to_{name_j}"
        return x, y, x_seg, y_seg, pair_id
