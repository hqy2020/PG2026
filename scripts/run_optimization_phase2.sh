#!/bin/bash
# Phase 2: Full-stack aggressive experiments with best-iteration tracking
PROJECT=/home/qyhu/Documents/r2_ours/PG2026
PYTHON=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
DATE=2026_05_30
mkdir -p $PROJECT/logs

# ─── Pancreas 2v: Full SPS+GAP+ADM, aggressive params ───
# Based on Exp-A + Exp-D, but with ALL modules enabled
# Key: higher density LR, lower grad threshold, GAP prune, ADM with low warmup
NAME_P1="${DATE}_pancreas_2views_fullstack_aggressive"
CMD_P1="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/pancreas_50_2views.pickle \
  -m output/$NAME_P1 \
  --ply_path data/234-sps/init_pancreas_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 5000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 25000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.03 --density_lr_final 0.003 \
  --save_iterations 5000 10000 15000 20000 25000 30000 35000 40000 \
  > logs/${NAME_P1}.log 2>&1 &"

# ─── Foot 2v: Full SPS+GAP+ADM, aggressive params ───
NAME_F1="${DATE}_foot_2views_fullstack_aggressive"
CMD_F1="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/foot_50_2views.pickle \
  -m output/$NAME_F1 \
  --ply_path data/234-sps/init_foot_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 5000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 25000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.03 --density_lr_final 0.003 \
  --save_iterations 5000 10000 15000 20000 25000 30000 35000 40000 \
  > logs/${NAME_F1}.log 2>&1 &"

echo "=== Launching Phase 2 ==="
eval $CMD_P1; PID_P1=$!
echo "GPU0 | Pancreas fullstack aggressive | PID=$PID_P1"
sleep 3
eval $CMD_F1; PID_F1=$!
echo "GPU1 | Foot fullstack aggressive | PID=$PID_F1"

echo ""
echo "Running experiments:"
echo "  GPU0: Exp-A (moderate) + Phase2-Pancreas (aggressive)"
echo "  GPU1: Exp-E (moderate) + Phase2-Foot (aggressive)"
echo ""
echo "Monitor: tail -f logs/${NAME_P1}.log"
echo "PID P1=$PID_P1 F1=$PID_F1"
