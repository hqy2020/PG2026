# XRA-GS: X-ray Reconstruction via Adaptive Gaussian Splatting

**Sparse-view CT Novel View Synthesis with 3D Gaussian Splatting**

> Pacific Graphics 2026 Submission

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE.md)

---

## Overview

**XRA-GS** is a novel Gaussian splatting framework for sparse-view CT novel view synthesis. It builds on R²-Gaussian (NeurIPS 2024) and introduces three novel components:

- **SPS** (Spatial Prior Seeding) — Density-weighted FDK initialization for better geometric prior
- **GAP** (Geometry-aware Pruning) — Proximity-guided densification in world coordinates
- **ADM** (Adaptive Density Modulation) — K-Planes based spatial density modulation

### Key Results (PSNR / SSIM)

| Setting | Metric | R²-Gaussian | XRA-GS (Ours) | Gain |
|---------|--------|-------------|--------------|------|
| **2-view** | PSNR ↑ | 21.34 | **22.41** | **+1.07** |
| **3-view** | PSNR ↑ | 27.83 | **28.22** | **+0.39** |
| **4-view** | PSNR ↑ | 29.18 | **29.30** | **+0.12** |

> XRA-GS achieves consistent gains across all sparsity levels, with the largest improvements at extreme sparsity (2-view +1.07 dB).

---

## Repository Structure

```
PG2026/
├── train.py                    # Training entry (method routing)
├── test.py                     # Evaluation
├── initialize_pcd.py           # SPS: point cloud initialization
├── r2_gaussian/                # Core Python package
│   ├── gaussian/               # XRA-GS / R²-Gaussian core
│   │   ├── gaussian_model.py   # GaussianModel class
│   │   ├── render_query.py     # Render/query functions
│   │   ├── kplanes.py          # K-Planes encoder (ADM)
│   │   └── initialize.py       # Initialization logic
│   ├── baselines/              # 6 comparison methods
│   │   ├── registry.py         # Method registry
│   │   ├── xgaussian/          # X-Gaussian
│   │   ├── fsgs/               # FSGS
│   │   ├── corgs/              # CoR-GS
│   │   ├── dngaussian/         # DN-Gaussian
│   │   ├── xfield/             # X-Field (NeurIPS 2025)
│   │   └── xgaussian/          # X-Gaussian
│   ├── innovations/            # Innovation modules
│   │   └── fsgs/               # Proximity densifier (GAP)
│   ├── dataset/                # Data loading
│   ├── utils/                  # Utilities
│   ├── arguments/              # CLI parameters
│   └── submodules/             # CUDA extensions
│       ├── simple-knn/         # KNN CUDA kernel
│       └── xray-gaussian-*/    # X-ray rasterization CUDA
├── docs/                       # Documentation
│   └── XRA_GS_GUIDE.md         # Paper writing guide
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

Download the dataset (contact authors) and place under `data/234/`:

```
data/234/
├── {organ}_50_{2|3|4}views.pickle
└── init_{organ}_50_{2|3|4}views.npy
```

### 3. Run Experiments

```bash
# Full XRA-GS
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_xrags \
  --ply_path data/234-sps/init_foot_50_2views.npy \
  --enable_fsgs_proximity --enable_adm --enable_gap

# Baseline (R²-Gaussian)
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_baseline

# Other 3DGS methods (via --method flag)
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_xgaussian --method xgaussian
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_fsgs --method fsgs
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_corgs --method corgs
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_dngaussian --method dngaussian
python train.py -s data/234/foot_50_2views.pickle \
  -m output/foot_2views_xfield --method xfield
```

### 4. Evaluation

```bash
python test.py -m output/<run_directory>
```

---

## XRA-GS Ablation Configurations

| Config | SPS | GAP | ADM | CLI flags |
|--------|-----|-----|-----|-----------|
| `baseline` | ✗ | ✗ | ✗ | *(none)* |
| `sps` | ✓ | ✗ | ✗ | `--enable_sps` |
| `gap` | ✗ | ✓ | ✗ | `--enable_gap` |
| `adm` | ✗ | ✗ | ✓ | `--enable_adm` |
| `sps_gap` | ✓ | ✓ | ✗ | `--enable_sps --enable_gap` |
| `sps_adm` | ✓ | ✗ | ✓ | `--enable_sps --enable_adm` |
| `gap_adm` | ✗ | ✓ | ✓ | `--enable_gap --enable_adm` |
| `xrags` (full) | ✓ | ✓ | ✓ | `--enable_fsgs_proximity --enable_gap --enable_adm` |

---

## Comparison Methods

| Method | Venue | Description |
|--------|-------|-------------|
| **R²-Gaussian** | NeurIPS 2024 | Radiative Gaussian Splatting (baseline) |
| **X-Gaussian** | ECCV 2024 | X-ray adapted 3DGS |
| **FSGS** | ECCV 2024 | Few-shot Gaussian Splatting |
| **CoR-GS** | ECCV 2024 | Co-regularized Gaussian Splatting |
| **DN-Gaussian** | CVPR 2024 | Depth-normalized sparse-view 3DGS |
| **X-Field** | NeurIPS 2025 | Implicit neural field with Gaussian densification |

---

## Citation

```bibtex
@inproceedings{xrags2026,
  title={XRA-GS: X-ray Reconstruction via Adaptive Gaussian Splatting 
         for Sparse-view CT Novel View Synthesis},
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
