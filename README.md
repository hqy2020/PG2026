# SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting

**Sparse-view CT Reconstruction with 3D Gaussian Splatting**

> Pacific Graphics 2026 Submission

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE.md)

---

## Overview

**SPAGS** is a spatial-aware progressive adaptive Gaussian splatting framework for sparse-view CT reconstruction. It builds on R²-Gaussian (NeurIPS 2024) and introduces three novel components:

- **SPS** (Spatial Prior Seeding) — Density-weighted FDK initialization for better geometric prior
- **GAP** (Geometry-aware Pruning) — Proximity-guided densification in world coordinates
- **ADM** (Adaptive Density Modulation) — K-Planes based spatial density modulation

### Key Results (PSNR / SSIM)

| Setting | Metric | R²-Gaussian | SPAGS (Ours) | Gain |
|---------|--------|-------------|--------------|------|
| **3-view** | PSNR ↑ | 27.88 | **28.35** | **+0.47** |
| **6-view** | PSNR ↑ | 33.18 | **33.40** | **+0.22** |
| **9-view** | PSNR ↑ | 36.09 | 36.03 | −0.06 |
| **9-view** | SSIM ↑ | 0.966 | **0.967** | +0.001 |

> SPAGS achieves the largest gains at extreme sparsity (3-view +0.47 dB).

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
│   ├── baselines/              # 5 comparison 3DGS methods
│   │   ├── registry.py         # Method registry
│   │   ├── xgaussian/          # X-Gaussian
│   │   ├── fsgs/               # FSGS
│   │   ├── dngaussian/         # DN-Gaussian
│   │   └── corgs/              # CoR-GS
│   ├── innovations/            # Innovation modules
│   │   └── fsgs/               # Proximity densifier (GAP)
│   ├── dataset/                # Data loading
│   ├── utils/                  # Utilities
│   ├── arguments/              # CLI parameters
│   └── submodules/             # CUDA extensions
│       ├── simple-knn/         # KNN CUDA kernel
│       └── xray-gaussian-*/    # X-ray rasterization CUDA
├── docs/                       # Documentation
│   └── SPAGS_PAPER_GUIDE.md    # Paper writing guide
└── scripts/                    # Utility scripts
```

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate r2_gaussian_new

# Build CUDA extensions
cd r2_gaussian/submodules/simple-knn && pip install -e .
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization && pip install -e .
```

### 2. Data Preparation

Download the dataset (contact authors) and place under `data/369/`:

```
data/369/
├── {organ}_50_{3|6|9}views.pickle
└── init_{organ}_50_{3|6|9}views.npy
```

### 3. Run Experiments

```bash
# Full SPAGS
python train.py -s data/369/foot_50_3views.pickle \
  -m output/2026_04_29_foot_3views_spags \
  --ply_path data/369/init_foot_50_3views.npy \
  --enable_sps --enable_gap --enable_adm

# Baseline (R²-Gaussian)
python train.py -s data/369/foot_50_3views.pickle \
  -m output/2026_04_29_foot_3views_baseline \
  --ply_path data/369/init_foot_50_3views.npy

# Other 3DGS methods (via --method flag)
python train.py -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_xgaussian --method xgaussian
python train.py -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_fsgs --method fsgs
python train.py -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_corgs --method corgs
python train.py -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_dngaussian --method dngaussian
```

### 4. Evaluation

```bash
python test.py -m output/<run_directory>
```

---

## SPAGS Ablation Configurations

| Config | SPS | GAP | ADM | CLI flags |
|--------|-----|-----|-----|-----------|
| `baseline` | ✗ | ✗ | ✗ | *(none)* |
| `sps` | ✓ | ✗ | ✗ | `--enable_sps` |
| `gap` | ✗ | ✓ | ✗ | `--enable_gap` |
| `adm` | ✗ | ✗ | ✓ | `--enable_adm` |
| `sps_gap` | ✓ | ✓ | ✗ | `--enable_sps --enable_gap` |
| `sps_adm` | ✓ | ✗ | ✓ | `--enable_sps --enable_adm` |
| `gap_adm` | ✗ | ✓ | ✓ | `--enable_gap --enable_adm` |
| `spags` | ✓ | ✓ | ✓ | `--enable_sps --enable_gap --enable_adm` |

---

## Comparison Methods

| Method | Venue | Description |
|--------|-------|-------------|
| **R²-Gaussian** | NeurIPS 2024 | Radiative Gaussian Splatting (baseline) |
| **X-Gaussian** | ECCV 2024 | X-ray adapted 3DGS |
| **FSGS** | ECCV 2024 | Few-shot Gaussian Splatting |
| **DN-Gaussian** | CVPR 2024 | Depth-normalized sparse-view 3DGS |
| **CoR-GS** | ECCV 2024 | Co-regularized Gaussian Splatting |

---

## Citation

```bibtex
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
and the [TIGRE toolbox](https://github.com/CERN/TIGRE.git).
