# NestedMorph: Deformable Medical Image Registration  

This repository implements the **NestedMorph** model, presented in the paper "[Kumar, Gurucharan Marthi Krishna, Mendola, Janine, Shmuel, Amir. "NestedMorph: Enhancing Deformable Medical Image Registration with Nested Attention Mechanisms." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Kumar_NestedMorph_Enhancing_Deformable_Medical_Image_Registration_with_Nested_Attention_Mechanisms_WACV_2025_paper.pdf)".

**[28.10.2024]**: ⭐ This paper was accepted to the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025).  
**[03.03.2025]**: 📢 This work was presented at WACV 2025 on March 3rd, 2025.  

[![arXiv](https://img.shields.io/badge/arXiv-2410.02550-b31b1b.svg)](https://arxiv.org/abs/2410.02550)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The NestedMorph framework combines advanced deep learning techniques for deformable medical image registration, leveraging Vision Transformer (ViT) and nested attention mechanisms to improve the accuracy and efficiency of image alignment.

## Model Comparison Table

| Model                | Architecture   | Parameters (Millions) |
|----------------------|----------------|-----------------------|
| VoxelMorph           | CNN            | 0.27                  |
| ViT-V-Net            | Transformer    | 31.33                 |
| TransMorph           | Transformer    | 46.75                 |
| **Proposed NestedMorph**      | **Transformer**    | **19.75**                |

⭐ The proposed **NestedMorph** is at least **57% less in size** than the current state-of-the-art Transformer models.

## Key Features  
- **3D Deformable Image Registration:** Designed for volumetric medical images to capture complex anatomical variations.  
- **Attention-Based Architecture:** Utilizes multi-scale attention mechanisms to enhance feature extraction and alignment.  
- **Spatial Transformer Integration:** Efficient warping of source images based on predicted deformation fields.  
- **Flexible Training:** Supports various loss functions (similarity metrics, regularization losses) and datasets.  
- **Optimized Input Size:** The study utilized images of size **64x64x64**, ensuring efficient training and inference.  

## Overall Framework  

The **Overall Framework** for NestedMorph combines multiple modules that work in synergy for medical image registration. The framework consists of the following key components:
- **Image Preprocessing:** Includes normalization, resizing, and augmentation of medical images.
- **Model Architecture:** Based on a Vision Transformer-based model with nested attention for improved feature extraction and registration accuracy.
- **Loss Functions:** Designed to optimize for both similarity metrics and geometric consistency between the source and target images.

![Overall Framework](Figures/OverallFramework.png)

## Model Architecture  
NestedMorph comprises:  
1. **Feature Extractor (Multi-Scale Encoder):** Extracts multi-scale features from input image pairs.
2. **Nested Attention Module:** Captures both local and global dependencies for enhanced feature aggregation.
3. **Decoder:** Refines and reconstructs the deformation field for precise alignment.
4. **Flow Estimation (Conv3D):** Predicts the deformation field for aligning the images.  
5. **Spatial Transformer:** Applies the deformation field to warp the source image.  

The model outputs:  
- **Registered Image:** The warped version of the source image, aligned to the target.  
- **Deformation Field:** Encodes the spatial transformation required for alignment.  

![NestedMorph Architecture](Figures/NestedMorph.png)

## Models  
This repository contains several models under the `Models/` directory. Each model can be used for deformable medical image registration. Below are the available models:

- [CycleMorph](src/models/cyclemorph/cycleMorph_model.py) - A model for cycle-consistent registration.
- [MIDIR](src/models/midir/midir.py) - Multi-scale Image Deformable Registration.
- [TransMorph](src/models/transmorph/TransMorph.py) - Transformer-based registration model.
- [ViT-V-Net](src/models/vitvnet/vitvnet.py) - Vision Transformer-based network for medical image registration.
- [VoxelMorph](src/models/voxelmorph.py) - A voxel-based deformable registration method.
- [NestedMorph](src/models/nestedmorph.py) - Proposed Nested multi-scale deformable model.

## Data Directory Structure  

The `data/` directory is structured as follows to organize training and testing datasets for deformable medical image registration:

### **Flowchart Diagram**

```
Registration_Data/
│
├── T1_Moving/                  # Directory for source images (e.g., T1-weighted MRI scans)
│   ├── subject1_t1.pkl         # Pickle files containing source images (e.g., T1 image for subject 1)
│   ├── subject2_t1.pkl         # Pickle files containing source images (e.g., T1 image for subject 2)
│   └── ...
│
├── Diffusion_Fixed/            # Directory for target images (e.g., DWI or atlas images)
│   ├── subject1_dwi.pkl        # Pickle files containing target images (e.g., DWI image for subject 1)
│   ├── subject2_dwi.pkl        # Pickle files containing target images (e.g., DWI image for subject 2)
│   └── ...
```

- **T1_Moving/**: Contains the source images for registration. These could be T1-weighted MRI scans or any other type of image used as the "moving" image.
- **Diffusion_Fixed/**: Contains the target images for registration. These could be Diffusion Weighted Imaging (DWI) scans or any other type of image used as the "fixed" image.

Each `subjectID_t1.pkl` file should match a corresponding `subjectID_dwi.pkl` file for proper pairing in the registration tasks.

## Installation  

### Prerequisites  
- Python 3.8+  
- CUDA-enabled GPU (optional but recommended)  
- Dependencies: PyTorch, NumPy, SciPy, Matplotlib  

### Setup  
1. Clone the repository:  
   ```bash  
   git clone [https://github.com/your-repo/NestedMorph.git](https://github.com/AS-Lab/Marthi-et-al-2024-NestedMorph-Deformable-Medical-Image-Registration)  
   cd Marthi-et-al-2024-NestedMorph-Deformable-Medical-Image-Registration  
   ```

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```

3. Training  
To train the model, run the following command with the appropriate parameters:
   ```bash  
   python main.py --t1_dir ./Registration_Data/T1_Moving/ --dwi_dir ./Registration_Data/Diffusion_Fixed/ --epochs 100 --img_size 64,64,64 --lr 2e-4 --batch_size 2 --cont_training --model_label NestedMorph  
   ```

## Registration Results  

This section presents the **Registration Results** achieved by the NestedMorph model on various datasets, including brain MRI and other medical image modalities. The results demonstrate the effectiveness of the NestedMorph model in performing deformable image registration, particularly in terms of accuracy and computational efficiency.

![Registration Results](Figures/Registration.png)

## Citation  

If you find this code useful in your research, please consider citing:

```
@inproceedings{kumar2025nestedmorph,
  title={NestedMorph: Enhancing Deformable Medical Image Registration with Nested Attention Mechanisms},
  author={Kumar, Gurucharan Marthi Krishna and Mendola, Janine and Shmuel, Amir},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```
