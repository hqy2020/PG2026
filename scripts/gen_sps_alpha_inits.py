#!/usr/bin/env python3
"""
Generate SPS init files for α (unif_ratio) sweep.
α values: 0.0, 0.1, 0.2, 0.5, 1.0  (0.2 already exists in data/234-sps)
"""
import os, sys, subprocess, numpy as np

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
PY = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA = f"{BASE}/data/234/chest_50_3views.pickle"
OUT_DIR = f"{BASE}/data"

os.chdir(BASE)

alpha_configs = [
    (0.0, 1.0, "alpha0.0"),      # pure density-weighted
    (0.1, 1.0, "alpha0.1"),       # mostly density-weighted
    # 0.2 already exists as data/234-sps/ (default)
    (0.5, 1.0, "alpha0.5"),       # balanced
    (1.0, 1.0, "alpha1.0"),       # pure uniform
]

for unif, gamma, label in alpha_configs:
    init_dir = f"{OUT_DIR}/sps-alpha/{label}"
    os.makedirs(init_dir, exist_ok=True)
    out_path = f"{init_dir}/init_chest_50_3views.npy"
    
    if os.path.exists(out_path):
        d = np.load(out_path)
        print(f"  ✅ {label} (unif={unif}, γ={gamma}): already exists ({d.shape[0]} pts)")
        continue
    
    cmd = [
        PY, "initialize_pcd.py", "--data", DATA,
        "--enable_sps", "--sps_strategy", "adaptive",
        "--sps_uniform_ratio", str(unif),
        "--sps_density_gamma", str(gamma),
        "--output", str(out_path)
    ]
    
    print(f"  🏗️  {label} (unif={unif}, γ={gamma})...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        d = np.load(out_path)
        print(f"✅ {d.shape[0]} pts")
    else:
        print(f"❌ {r.stderr[:200]}")

print("\n✅ All SPS init files generated!")
