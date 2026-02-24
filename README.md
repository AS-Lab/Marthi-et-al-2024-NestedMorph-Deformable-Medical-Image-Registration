# NestedMorph: Deformable Medical Image Registration  

This repository implements the **NestedMorph** model, presented in the paper "[Kumar, Gurucharan Marthi Krishna, Mendola, Janine, Shmuel, Amir. "NestedMorph: Enhancing Deformable Medical Image Registration with Nested Attention Mechanisms." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Kumar_NestedMorph_Enhancing_Deformable_Medical_Image_Registration_with_Nested_Attention_Mechanisms_WACV_2025_paper.pdf)".

**[28.10.2024]**: ⭐ This paper was accepted to the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025).  
**[03.03.2025]**: 📢 This work was presented at WACV 2025 on March 3rd, 2025.  

[![arXiv](https://img.shields.io/badge/arXiv-2410.02550-b31b1b.svg)](https://arxiv.org/abs/2410.02550)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Available Models](#available-models)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [Custom T1/DWI Dataset](#custom-t1dwi-dataset)
  - [IXI Brain MRI Dataset](#ixi-brain-mri-dataset)
  - [OASIS Brain MRI Dataset](#oasis-brain-mri-dataset)
  - [Data Format Conversion](#data-format-conversion)
- [Training](#training)
  - [Custom Dataset](#training-on-a-custom-dataset)
  - [IXI Dataset](#training-on-ixi)
  - [OASIS Dataset](#training-on-oasis)
  - [CycleMorph](#training-cyclemorph)
  - [Resuming Training](#resuming-training)
- [Inference](#inference)
  - [IXI Inference](#ixi-inference)
  - [OASIS Inference](#oasis-inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contact](#contact)

---

The NestedMorph framework combines advanced deep learning techniques for deformable medical image registration, leveraging Vision Transformer (ViT) and nested attention mechanisms to improve the accuracy and efficiency of image alignment.

## Model Comparison Table

| Model                | Architecture   | Parameters (Millions) |
|----------------------|----------------|-----------------------|
| VoxelMorph           | CNN            | 0.27                  |
| ViT-V-Net            | Transformer    | 31.33                 |
| TransMorph           | Transformer    | 46.75                 |
| **Proposed NestedMorph**      | **Transformer**    | **21.95**                |

⭐ NestedMorph is at least **53% smaller** than prior Transformer baselines while matching or exceeding their registration accuracy.

**Key features:**
- 🧠 **3D Deformable Registration** for volumetric medical images
- 👁️ **Nested Attention** integrating encoder and decoder attention weights for efficient spatial feature aggregation
- 🔄 **Multi-Scale Encoder/Decoder** with deformable flow estimation
- ⚙️ **Flexible** — supports NCC + diffusion regularization, customisable image sizes, `.pkl` and `.nii.gz` data formats
- 📦 **Multiple baselines included** for fair comparison (VoxelMorph, TransMorph, ViT-V-Net, MIDIR, CycleMorph)

---

## Architecture

### Overall Framework

The framework covers image preprocessing, a ViT-based encoder with nested attention, a decoder for deformation field prediction, and a spatial transformer for warping.

![Overall Framework](Figures/OverallFramework.png)

### NestedMorph Architecture

1. **Multi-Scale Encoder** — extracts hierarchical features from the concatenated moving/fixed image pair
2. **Nested Attention Module** — fuses encoder and decoder attention weights to capture both local and global dependencies
3. **Decoder** — upsamples and refines the deformation field
4. **Flow Estimation (Conv3D)** — predicts the dense 3D deformation field
5. **Spatial Transformer** — applies the deformation field to warp the moving image

Outputs: **(1) registered image**, **(2) deformation field**

![NestedMorph Architecture](Figures/NestedMorph.png)

---

## Available Models

All models live under `src/models/` and share the same training interface.

| Model | File | Type |
|---|---|---|
| **NestedMorph** | `src/models/nestedmorph.py` | Transformer + Nested Attention (proposed) |
| VoxelMorph | `src/models/voxelmorph.py` | CNN |
| TransMorph | `src/models/transmorph/TransMorph.py` | Swin Transformer |
| ViT-V-Net | `src/models/vitvnet/vitvnet.py` | Vision Transformer |
| MIDIR | `src/models/midir/midir.py` | B-Spline CNN |
| CycleMorph | `src/models/cyclemorph/` | Cycle-consistent CNN |

Valid `--model_label` values: `NestedMorph`, `VoxelMorph`, `TransMorph`, `ViTVNet`, `MIDIR`, `CycleMorph`

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch 1.9+

### Setup

```bash
git clone https://github.com/AS-Lab/Marthi-et-al-2024-NestedMorph-Deformable-Medical-Image-Registration
cd Marthi-et-al-2024-NestedMorph-Deformable-Medical-Image-Registration
pip install -r requirements.txt
pip install nibabel  # required for .nii.gz support (not in requirements.txt)
```

> **Note:** `nibabel` is used by the IXI/OASIS dataset loaders and all inference scripts. Install it separately as shown above.

---

## Data Preparation

All datasets support **`.pkl`** (TransMorph-style pickle) or **`.nii.gz`** (NIfTI) format. Both are handled automatically. Use `scripts/convert_data.py` to convert between formats.

### Custom T1/DWI Dataset

For paired T1 (moving) and DWI/diffusion (fixed) registration, organize your data as:

```
Registration_Data/
├── T1_Moving/
│   ├── HCA6002236_V1_MR_T1w_MPR_vNav.nii.gz
│   ├── HCA6010538_V1_MR_T1w_MPR_vNav.nii.gz
│   └── ...
└── Diffusion_Fixed/
    ├── HCA6002236_V1_MR_dir99_AP.nii.gz
    ├── HCA6010538_V1_MR_dir99_AP.nii.gz
    └── ...
```

**Automatic pairing:** the loader extracts the subject ID from the first `_`-delimited token of each filename and pairs files with matching IDs. An 80/20 train/val split is applied automatically.

---

### IXI Brain MRI Dataset

IXI uses **atlas-to-patient** registration (one fixed atlas, many moving patient scans). Available from [brain-development.org](http://brain-development.org/ixi-dataset/) or in TransMorph-preprocessed `.pkl` format.

Expected directory structure:

```
data/IXI/
├── atlas.pkl              # fixed atlas volume (.pkl or .nii.gz)
├── Train/
│   ├── subject_001.pkl
│   ├── subject_001_seg.pkl    # optional segmentation labels
│   └── ...
├── Val/
│   └── ...
└── Test/
    └── ...
```

Default split: **403 train / 58 val / 115 test**  
Default image size: **160 × 192 × 224**  
Config: `configs/ixi.yaml`

---

### OASIS Brain MRI Dataset

OASIS uses **inter-patient** registration (random pairs drawn from the dataset). Available from [oasis-brains.org](https://www.oasis-brains.org/) or the [Learn2Reg challenge (Task 03)](https://learn2reg.grand-challenge.org/).

Expected directory structure:

```
data/OASIS/
├── Train/
│   ├── OASIS_0001.pkl
│   ├── OASIS_0001_seg.pkl     # optional segmentation labels
│   └── ...
├── Val/
│   └── ...
└── Test/
    └── ...
```

Default split: **394 train / 19 val / 38 test**  
Default image size: **160 × 192 × 224**  
Config: `configs/oasis.yaml`

---

### Data Format Conversion

```bash
# Convert a directory of .pkl files → .nii.gz
python scripts/convert_data.py \
    --input_dir data/IXI/Train \
    --output_dir data/IXI_nifti/Train \
    --to nii.gz

# Convert a directory of .nii.gz files → .pkl
python scripts/convert_data.py \
    --input_dir data/IXI_nifti/Train \
    --output_dir data/IXI_pkl/Train \
    --to pkl

# Convert a single file
python scripts/convert_data.py \
    --input_file data/IXI/atlas.pkl \
    --output_dir data/IXI_nifti/ \
    --to nii.gz
```

---

## Training

### Training on a Custom Dataset

Use `main.py` for any paired moving/fixed dataset (e.g. T1/DWI):

```bash
python main.py \
    --t1_dir ./Registration_Data/T1_Moving/ \
    --dwi_dir ./Registration_Data/Diffusion_Fixed/ \
    --model_label NestedMorph \
    --epochs 500 \
    --img_size 64,64,64 \
    --lr 2e-4 \
    --batch_size 2
```

Checkpoints → `experiments/<model_label>_1_ncc_1_diffusion_1/`  
Logs → `logs/<model_label>_1_ncc_1_diffusion_1/`

---

### Training on IXI

```bash
python scripts/train_ixi.py \
    --data_dir ./data/IXI \
    --atlas_path ./data/IXI/atlas.pkl \
    --model_label NestedMorph \
    --epochs 500 \
    --img_size 160,192,224 \
    --lr 0.0001 \
    --batch_size 1
```

Checkpoints → `experiments/IXI_NestedMorph_ncc_1_diffusion_1/`

---

### Training on OASIS

```bash
python scripts/train_oasis.py \
    --data_dir ./data/OASIS \
    --model_label NestedMorph \
    --epochs 500 \
    --img_size 160,192,224 \
    --lr 0.0001 \
    --batch_size 1
```

Checkpoints → `experiments/OASIS_NestedMorph_ncc_1_diffusion_1/`

---

### Training CycleMorph

CycleMorph has its own training loop and is dispatched automatically via `main.py`:

```bash
python main.py \
    --t1_dir ./Registration_Data/T1_Moving/ \
    --dwi_dir ./Registration_Data/Diffusion_Fixed/ \
    --model_label CycleMorph \
    --epochs 500 \
    --img_size 64,64,64 \
    --lr 1e-4 \
    --batch_size 1
```

---

### Resuming Training

Pass `--cont_training` to any script to load the latest checkpoint and continue from there:

```bash
python scripts/train_ixi.py \
    --data_dir ./data/IXI \
    --atlas_path ./data/IXI/atlas.pkl \
    --model_label NestedMorph \
    --epochs 500 \
    --cont_training
```

---

## Inference

Inference scripts output warped images, deformation flow fields, and (if segmentation files are present) warped segmentations — all as `.nii.gz`.

### IXI Inference

```bash
python scripts/infer_ixi.py \
    --data_dir ./data/IXI \
    --atlas_path ./data/IXI/atlas.pkl \
    --model_label NestedMorph \
    --checkpoint experiments/IXI_NestedMorph_ncc_1_diffusion_1/final_model_NestedMorph.pth.tar \
    --img_size 160,192,224 \
    --output_dir results/IXI/NestedMorph
```

### OASIS Inference

```bash
python scripts/infer_oasis.py \
    --data_dir ./data/OASIS \
    --model_label NestedMorph \
    --checkpoint experiments/OASIS_NestedMorph_ncc_1_diffusion_1/final_model_NestedMorph.pth.tar \
    --img_size 160,192,224 \
    --output_dir results/OASIS/NestedMorph
```

Output layout (same for both):

```
results/<dataset>/<model>/
├── warped/          # registered moving images (.nii.gz)
├── flow/            # deformation fields (.nii.gz, shape H×W×D×3)
└── warped_seg/      # warped segmentations (.nii.gz, if labels available)
```

---

## Evaluation

`scripts/evaluate.py` computes four metrics on the test set and supports multi-model comparison:

| Metric | Description |
|---|---|
| **DSC** | Dice Similarity Coefficient per anatomical label (requires segmentations) |
| **SSIM** | 3D Structural Similarity Index |
| **HD95** | 95th percentile Hausdorff Distance in mm (requires segmentations) |
| **\|Jac\|≤0 (%)** | Percentage of voxels with non-positive Jacobian determinant (topology violations) |

```bash
# Evaluate NestedMorph on IXI
python scripts/evaluate.py \
    --dataset ixi \
    --data_dir ./data/IXI \
    --atlas_path ./data/IXI/atlas.pkl \
    --model_label NestedMorph \
    --checkpoint experiments/IXI_NestedMorph_ncc_1_diffusion_1/final_model_NestedMorph.pth.tar

# Compare multiple models on OASIS
python scripts/evaluate.py \
    --dataset oasis \
    --data_dir ./data/OASIS \
    --model_label NestedMorph VoxelMorph TransMorph \
    --checkpoint ckpt_nested.pth.tar ckpt_voxel.pth.tar ckpt_trans.pth.tar
```

Per-subject CSVs → `results/<dataset>_<model>_eval.csv`  
Summary comparison table → `results/<dataset>_comparison.csv`

---

## Results

![Registration Results](Figures/Registration.png)

---

## Project Structure

```
.
├── main.py                        # Entry point for custom T1/DWI training
├── requirements.txt
├── configs/
│   ├── ixi.yaml                   # IXI dataset & training config
│   └── oasis.yaml                 # OASIS dataset & training config
├── scripts/
│   ├── train.py                   # Training loop (custom dataset)
│   ├── train_ixi.py               # Training loop (IXI atlas-to-patient)
│   ├── train_oasis.py             # Training loop (OASIS inter-patient)
│   ├── train_cyclemorph.py        # CycleMorph-specific training loop
│   ├── infer_ixi.py               # IXI inference → .nii.gz outputs
│   ├── infer_oasis.py             # OASIS inference → .nii.gz outputs
│   ├── evaluate.py                # DSC / SSIM / HD95 / Jacobian evaluation
│   └── convert_data.py            # .pkl ↔ .nii.gz conversion utility
├── src/
│   ├── data/
│   │   ├── datasets.py            # RegistrationDataset (paired T1/DWI, pkl + nii.gz)
│   │   ├── ixi_dataset.py         # IXIBrainDataset / IXIBrainInferDataset
│   │   ├── oasis_dataset.py       # OASISBrainDataset / OASISBrainInferDataset
│   │   ├── data_utils.py          # pkload and helpers
│   │   ├── trans.py               # Spatial transforms
│   │   └── rand.py                # Random augmentations (RandomFlip, NumpyType)
│   ├── losses/
│   │   └── losses.py              # NCC_vxm, Grad3d
│   ├── models/
│   │   ├── nestedmorph.py         # NestedMorph (proposed)
│   │   ├── voxelmorph.py          # VoxelMorph baseline
│   │   ├── transmorph/            # TransMorph baseline
│   │   ├── vitvnet/               # ViT-V-Net baseline
│   │   ├── midir/                 # MIDIR B-Spline baseline
│   │   └── cyclemorph/            # CycleMorph baseline
│   └── utils/
│       ├── config.py              # Device (CPU/GPU) config
│       └── utils.py               # AverageMeter, register_model, similarity, etc.
└── notebooks/
    └── demo.ipynb                 # Interactive demo notebook
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{kumar2025nestedmorph,
  title={NestedMorph: Enhancing Deformable Medical Image Registration with Nested Attention Mechanisms},
  author={Kumar, Gurucharan Marthi Krishna and Mendola, Janine and Shmuel, Amir},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

---

## Contact

- **Gurucharan Marthi Krishna Kumar**: [gurucharan.marthikrishnakumar@mail.mcgill.ca](mailto:gurucharan.marthikrishnakumar@mail.mcgill.ca)
- **Issues / Questions**: [GitHub Issues](https://github.com/AS-Lab/Marthi-et-al-2024-NestedMorph-Deformable-Medical-Image-Registration/issues)

---

<div align="center">

**🌟 Star us on GitHub — it helps!**

[⬆ Back to Top](#nestedmorph-deformable-medical-image-registration)

</div>
