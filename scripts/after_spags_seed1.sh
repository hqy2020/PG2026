#!/bin/bash
# Evaluate SPAGS stability runs and launch X-Field experiments
set -e

PYTHON=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026

# 1. Evaluate SPAGS seed1 (GPU0 will free up first)
echo "Evaluating chest_stability_spags_seed1..."
CUDA_VISIBLE_DEVICES=0 $PYTHON $BASE/test.py -m $BASE/output/chest_stability_spags_seed1 --iteration 30000 2>&1 | tail -5

# 2. Start X-Field chest_2views on GPU0
echo "Starting X-Field chest_2views on GPU0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON /home/qyhu/Documents/r2_ours/X-Field/train.py \
  -s $BASE/data/234/chest_50_2views.pickle \
  -m $BASE/output/xfield/chest_2views_xfield \
  --ply_path $BASE/data/234-sps/init_chest_50_2views.npy \
  --config /home/qyhu/Documents/r2_ours/X-Field/configs/xfield_30000.yaml \
  --test_iterations 30000 --save_iterations 30000 --quiet \
  2>&1 | tail -5
