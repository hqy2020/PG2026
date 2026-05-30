#!/bin/bash
# Plan B: Alternative strategies for 2-view P0 optimization
# Key ideas:
# 1. Lower TV regularization (lambda_tv=0.01, lambda_plane_tv=0.0001)
# 2. No ADM (avoids k-planes smoothing)
# 3. No SPS init (random init like X-Gaussian)
# 4. Much higher density LR + lower grad threshold

PROJECT=/home/qyhu/Documents/r2_ours/PG2026
PYTHON=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
DATE=2026_05_30
mkdir -p $PROJECT/logs

# ─── Pancreas 2v: low TV, no SPS, dense ───
# Idea: random init + aggressive densification + low TV + low ADM TV
CMD_P1="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/pancreas_50_2views.pickle \
  -m output/${DATE}_pancreas_2views_planB_lowtv_nosps \
  --lambda_tv 0.01 \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 5000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --lambda_plane_tv 0.0001 \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 25000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.0001 \
  --density_lr_init 0.05 --density_lr_final 0.005 \
  --position_lr_init 0.0003 --position_lr_final 0.00003 \
  > logs/${DATE}_pancreas_2views_planB_lowtv_nosps.log 2>&1 &"

# ─── Foot 2v: same approach ───
CMD_F1="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/foot_50_2views.pickle \
  -m output/${DATE}_foot_2views_planB_lowtv_nosps \
  --lambda_tv 0.01 \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 5000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --lambda_plane_tv 0.0001 \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 25000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.0001 \
  --density_lr_init 0.05 --density_lr_final 0.005 \
  --position_lr_init 0.0003 --position_lr_final 0.00003 \
  > logs/${DATE}_foot_2views_planB_lowtv_nosps.log 2>&1 &"

echo "=== Plan B: Ready to launch ==="
echo "GPU0: Pancreas 2v - no SPS init, low TV, high density LR"
echo "GPU1: Foot 2v - no SPS init, low TV, high density LR"
echo ""
echo "To launch: bash scripts/run_planB.sh"
echo "Check GPU availability first with nvidia-smi"
