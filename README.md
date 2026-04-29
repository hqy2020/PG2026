# SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting

**Sparse-view CT Reconstruction with 3D Gaussian Splatting**

> Pacific Graphics 2026 Submission

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE.md)

---

## Overview

**SPAGS** is a spatial-aware progressive adaptive Gaussian splatting framework for sparse-view CT reconstruction. It builds on R²-Gaussian (NeurIPS 2024) and introduces three novel components:

- **SPS** (Spatial Prior Seeding) — Density-weighted FDK initialization for better geometric prior
- **GAR** (Geometry-aware Refinement) — Proximity-guided densification in world coordinates
- **ADM** (Adaptive Density Modulation) — K-Planes based spatial density modulation

![SPAGS Framework](docs/fig4-1.pdf)

### Key Results (PSNR / SSIM)

| Setting | Metric | R²-Gaussian | SPAGS (Ours) | Gain |
|---------|--------|-------------|--------------|------|
| **3-view** | PSNR ↑ | 27.88 | **28.35** | **+0.47** |
| **3-view** | SSIM ↑ | 0.903 | **0.906** | +0.003 |
| **6-view** | PSNR ↑ | 33.18 | **33.40** | **+0.22** |
| **6-view** | SSIM ↑ | 0.952 | **0.954** | +0.002 |
| **9-view** | PSNR ↑ | 36.09 | **36.03** | −0.06 |
| **9-view** | SSIM ↑ | 0.966 | **0.967** | +0.001 |

> SPAGS achieves consistent improvements in the most challenging sparse-view settings (3/6 views), with the largest gains at extreme sparsity.

---

## Repository Structure

```
PG2026/
├── train.py                    # Training entry (method routing)
├── test.py                     # Evaluation
├── initialize_pcd.py           # SPS: point cloud initialization
├── r2_gaussian/                # Core Python package
│   ├── gaussian/               # SPAGS / R²-Gaussian core
│   │   ├── gaussian_model.py   # GaussianModel class
│   │   ├── render_query.py     # Render/query functions
│   │   ├── kplanes.py          # K-Planes encoder (ADM)
│   │   └── initialize.py       # Initialization logic
│   ├── baselines/              # 7 comparison methods
│   │   ├── registry.py         # Method registry
│   │   ├── xgaussian/          # X-Gaussian
│   │   ├── fsgs/               # FSGS
│   │   ├── dngaussian/         # DN-Gaussian
│   │   ├── corgs/              # CoR-GS
│   │   ├── naf/                # NAF (NeRF)
│   │   ├── tensorf/            # TensoRF (NeRF)
│   │   └── saxnerf/            # SAX-NeRF (NeRF)
│   ├── innovations/            # Innovation modules
│   │   └── fsgs/               # Proximity densifier (GAR)
│   ├── dataset/                # Data loading
│   ├── utils/                  # Utilities
│   ├── arguments/              # CLI parameters
│   └── submodules/             # CUDA extensions
│       ├── simple-knn/         # KNN CUDA kernel
│       └── xray-gaussian-*/    # X-ray rasterization CUDA
├── cc-agent/                   # AI research assistant system
│   ├── scripts/                # Experiment scripts
│   │   └── run_spags_ablation.sh  # Main ablation script
│   └── experiment/             # Experiment results
├── docs/                       # Documentation
│   └── chapter4_thesis.tex     # Thesis Chapter 4 (Chinese)
└── data_generator/             # Data generation tools
```

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate r2_gaussian_new

# Build CUDA extensions
cd r2_gaussian/submodules/simple-knn && pip install -e .
cd ../../../
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization && pip install -e .
cd ../../../

# Install TIGRE for FDK initialization (optional, for new data)
pip install TIGRE-2.3/Python --no-build-isolation
```

### 2. Data Preparation

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1YZ3w87XrCNyjDRos6gkY8zgT5hESl-PN?usp=sharing) and organize as:

```
data/369/
├── chest_50_3views.pickle
├── chest_50_6views.pickle
├── foot_50_3views.pickle
├── foot_50_6views.pickle
├── head_50_9views.pickle
├── ...
├── init_chest_50_3views.npy
├── init_foot_50_3views.npy
└── ...
```

### 3. Run Experiments

```bash
# Full SPAGS
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0

# Baseline (R²-Gaussian)
./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1

# Other comparison methods
./cc-agent/scripts/run_spags_ablation.sh xgaussian foot 3 0
./cc-agent/scripts/run_spags_ablation.sh fsgs chest 6 1
./cc-agent/scripts/run_spags_ablation.sh dngaussian head 9 0
./cc-agent/scripts/run_spags_ablation.sh corgs abdomen 3 0

# NeRF methods
./cc-agent/scripts/run_spags_ablation.sh naf chest 6 1
./cc-agent/scripts/run_spags_ablation.sh tensorf head 9 0
./cc-agent/scripts/run_spags_ablation.sh saxnerf abdomen 3 0

# Ablation variants
./cc-agent/scripts/run_spags_ablation.sh sps foot 3 0
./cc-agent/scripts/run_spags_ablation.sh gar foot 3 0
./cc-agent/scripts/run_spags_ablation.sh adm foot 3 0
```

### 4. Evaluation

```bash
python test.py -m output/<run_directory>
```

---

## SPAGS Ablation Configurations

| Config | SPS | GAR | ADM | Description |
|--------|-----|-----|-----|-------------|
| `baseline` | ✗ | ✗ | ✗ | R²-Gaussian baseline |
| `sps` | ✓ | ✗ | ✗ | Spatial Prior Seeding only |
| `gar` | ✗ | ✓ | ✗ | Geometry-aware Refinement only |
| `adm` | ✗ | ✗ | ✓ | Adaptive Density Modulation only |
| `sps_gar` | ✓ | ✓ | ✗ | SPS + GAR |
| `sps_adm` | ✓ | ✗ | ✓ | SPS + ADM |
| `gar_adm` | ✗ | ✓ | ✓ | GAR + ADM |
| `spags` | ✓ | ✓ | ✓ | Full SPAGS |

---

## Citation

If you find this work useful, please cite:

```
@inproceedings{spags2026,
  title={SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting 
         for Sparse-view CT Reconstruction},
  author={...},
  booktitle={Pacific Graphics},
  year={2026}
}
```

---

## Acknowledgement

This code is built upon [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) (NeurIPS 2024), 
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), 
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF), 
[NAF](https://github.com/Ruyi-Zha/naf_cbct), 
and the [TIGRE toolbox](https://github.com/CERN/TIGRE.git).
