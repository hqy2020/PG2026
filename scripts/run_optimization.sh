#!/bin/bash
# 2-view P0 optimization experiments - Staggered 2 at a time
PROJECT=/home/qyhu/Documents/r2_ours/PG2026
PYTHON=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
DATE=2026_05_30
mkdir -p $PROJECT/logs

# ─── PHASE 1: Launch 2 Pancreas + 2 Foot ───
echo "=== PHASE 1: Launching initial batch ==="

# Exp-A: Pancreas 2v - GAP prune + ADM + proximity densifier (aggressive)
NAME_A="${DATE}_pancreas_2views_opt_A_gap_adm"
CMD_A="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py -s data/234/pancreas_50_2views.pickle -m output/$NAME_A \
  --ply_path data/234-sps/init_pancreas_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 8000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 20000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.02 --density_lr_final 0.002 > logs/${NAME_A}.log 2>&1 &"

# Exp-D: Pancreas 2v - No ADM, aggressive proximity + GAP prune
NAME_D="${DATE}_pancreas_2views_opt_D_prune_aggressive"
CMD_D="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py -s data/234/pancreas_50_2views.pickle -m output/$NAME_D \
  --ply_path data/234-sps/init_pancreas_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.1 --proximity_k_neighbors 3 \
  --proximity_start_iter 500 --proximity_interval 200 --proximity_until_iter 20000 \
  --enable_gap --gap_k 3 --gap_threshold 0.005 --gap_max_ratio 0.02 \
  --gap_start_iter 800 --gap_interval 300 --gap_until_iter 15000 \
  --iterations 40000 --densify_until_iter 25000 --densify_grad_threshold 0.0001 \
  --density_lr_init 0.03 --density_lr_final 0.003 > logs/${NAME_D}.log 2>&1 &"

# Exp-E: Foot 2v - GAP prune + ADM (like Exp-A)
NAME_E="${DATE}_foot_2views_opt_E_gap_adm"
CMD_E="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py -s data/234/foot_50_2views.pickle -m output/$NAME_E \
  --ply_path data/234-sps/init_foot_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 8000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 20000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.02 --density_lr_final 0.002 > logs/${NAME_E}.log 2>&1 &"

# Exp-F: Foot 2v - No ADM, aggressive prune
NAME_F="${DATE}_foot_2views_opt_F_prune_aggressive"
CMD_F="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py -s data/234/foot_50_2views.pickle -m output/$NAME_F \
  --ply_path data/234-sps/init_foot_50_2views.npy \
  --enable_fsgs_proximity --proximity_threshold 0.1 --proximity_k_neighbors 3 \
  --proximity_start_iter 500 --proximity_interval 200 --proximity_until_iter 20000 \
  --enable_gap --gap_k 3 --gap_threshold 0.005 --gap_max_ratio 0.02 \
  --gap_start_iter 800 --gap_interval 300 --gap_until_iter 15000 \
  --iterations 40000 --densify_until_iter 25000 --densify_grad_threshold 0.0001 \
  --density_lr_init 0.03 --density_lr_final 0.003 > logs/${NAME_F}.log 2>&1 &"

eval $CMD_A; PID_A=$!; echo "GPU0 | Exp-A ($NAME_A) | PID=$PID_A"
sleep 3
eval $CMD_D; PID_D=$!; echo "GPU0 | Exp-D ($NAME_D) | PID=$PID_D"
sleep 3
eval $CMD_E; PID_E=$!; echo "GPU1 | Exp-E ($NAME_E) | PID=$PID_E"
sleep 3
eval $CMD_F; PID_F=$!; echo "GPU1 | Exp-F ($NAME_F) | PID=$PID_F"

echo ""
echo "=== PHASE 1 launched. Waiting for completion (~20-40 min) ==="
echo "Check: tail -f logs/${NAME_A}.log"
echo "Check: tail -f logs/${NAME_E}.log"
echo "PIDs: A=$PID_A D=$PID_D E=$PID_E F=$PID_F"
