#!/bin/bash
# 全量 P0 + P1 优化实验 — 一口气跑完
# 所有实验使用完整 SPS+GAP+ADM 三件套，取最佳迭代 PSNR
PROJECT=/home/qyhu/Documents/r2_ours/PG2026
PYTHON=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
DATE=2026_05_30
mkdir -p $PROJECT/logs

# ─── 通用配置模板 ───
BASE_P0="--enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 500 --proximity_interval 300 --proximity_until_iter 25000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 5000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.008 --gap_max_ratio 0.01 \
  --gap_start_iter 1000 --gap_interval 500 --gap_until_iter 25000 \
  --iterations 40000 --densify_until_iter 30000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.03 --density_lr_final 0.003"

BASE_P1="--enable_fsgs_proximity --proximity_threshold 0.05 --proximity_k_neighbors 5 \
  --proximity_start_iter 1000 --proximity_interval 500 --proximity_until_iter 15000 \
  --enable_adm --adm_resolution 64 --adm_feature_dim 32 --adm_max_range 0.3 \
  --adm_warmup_iters 6000 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --enable_gap --gap_k 3 --gap_threshold 0.01 --gap_max_ratio 0.01 \
  --gap_start_iter 2000 --gap_interval 500 --gap_until_iter 20000 \
  --iterations 35000 --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --density_lr_init 0.015 --density_lr_final 0.0015"

# ─── P0: Pancreas 2v ───
# Plan B: low TV + no SPS init (already have Exp-A + Phase2 running)
NAME_PB="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/pancreas_50_2views.pickle \
  -m output/${DATE}_pancreas_2views_planB_lowtv \
  --lambda_tv 0.01 \
  $BASE_P0 \
  > logs/${DATE}_pancreas_2views_planB_lowtv.log 2>&1 &"

# ─── P0: Foot 2v ───
# Plan B: low TV + no SPS init (already have Exp-E + Phase2 running)
NAME_FB="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/foot_50_2views.pickle \
  -m output/${DATE}_foot_2views_planB_lowtv \
  --lambda_tv 0.01 \
  $BASE_P0 \
  > logs/${DATE}_foot_2views_planB_lowtv.log 2>&1 &"

# ─── P1: Chest 2v ───
NAME_C2="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/chest_50_2views.pickle \
  -m output/${DATE}_chest_2views_p1 \
  --ply_path data/234-sps/init_chest_50_2views.npy \
  $BASE_P1 \
  > logs/${DATE}_chest_2views_p1.log 2>&1 &"

# ─── P1: Chest 4v ───
NAME_C4="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/chest_50_4views.pickle \
  -m output/${DATE}_chest_4views_p1 \
  --ply_path data/234-sps/init_chest_50_4views.npy \
  $BASE_P1 \
  > logs/${DATE}_chest_4views_p1.log 2>&1 &"

# ─── P1: Head 2v ───
NAME_H2="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/head_50_2views.pickle \
  -m output/${DATE}_head_2views_p1 \
  --ply_path data/234-sps/init_head_50_2views.npy \
  $BASE_P1 \
  > logs/${DATE}_head_2views_p1.log 2>&1 &"

# ─── P1: Head 3v ───
NAME_H3="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/head_50_3views.pickle \
  -m output/${DATE}_head_3views_p1 \
  --ply_path data/234-sps/init_head_50_3views.npy \
  $BASE_P1 \
  > logs/${DATE}_head_3views_p1.log 2>&1 &"

# ─── P1: Head 4v ───
NAME_H4="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/head_50_4views.pickle \
  -m output/${DATE}_head_4views_p1 \
  --ply_path data/234-sps/init_head_50_4views.npy \
  $BASE_P1 \
  > logs/${DATE}_head_4views_p1.log 2>&1 &"

# ─── P1: Foot 3v ───
NAME_F3="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/foot_50_3views.pickle \
  -m output/${DATE}_foot_3views_p1 \
  --ply_path data/234-sps/init_foot_50_3views.npy \
  $BASE_P1 \
  > logs/${DATE}_foot_3views_p1.log 2>&1 &"

# ─── P1: Foot 4v ───
NAME_F4="CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
  -s data/234/foot_50_4views.pickle \
  -m output/${DATE}_foot_4views_p1 \
  --ply_path data/234-sps/init_foot_50_4views.npy \
  $BASE_P1 \
  > logs/${DATE}_foot_4views_p1.log 2>&1 &"

# ─── P1: Pancreas 4v ───
NAME_P4="CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
  -s data/234/pancreas_50_4views.pickle \
  -m output/${DATE}_pancreas_4views_p1 \
  --ply_path data/234-sps/init_pancreas_50_4views.npy \
  $BASE_P1 \
  > logs/${DATE}_pancreas_4views_p1.log 2>&1 &"

# ─── 全部启动 ───
echo "=== 启动全部实验 ==="
echo "[GPU0]"

eval $NAME_PB;   PID1=$!;   echo "  Plan B Pancreas 2v    | PID=$PID1"
sleep 3
eval $NAME_C2;   PID2=$!;   echo "  P1 Chest 2v           | PID=$PID2"
sleep 3
eval $NAME_H2;   PID3=$!;   echo "  P1 Head 2v            | PID=$PID3"
sleep 3
eval $NAME_H4;   PID4=$!;   echo "  P1 Head 4v            | PID=$PID4"
sleep 3
eval $NAME_F4;   PID5=$!;   echo "  P1 Foot 4v            | PID=$PID5"
sleep 3

echo ""
echo "[GPU1]"
eval $NAME_FB;   PID6=$!;   echo "  Plan B Foot 2v        | PID=$PID6"
sleep 3
eval $NAME_C4;   PID7=$!;   echo "  P1 Chest 4v           | PID=$PID7"
sleep 3
eval $NAME_H3;   PID8=$!;   echo "  P1 Head 3v            | PID=$PID8"
sleep 3
eval $NAME_F3;   PID9=$!;   echo "  P1 Foot 3v            | PID=$PID9"
sleep 3
eval $NAME_P4;   PID10=$!;  echo "  P1 Pancreas 4v        | PID=$PID10"

echo ""
echo "=== 全部已启动 ==="
echo "GPU0: Exp-A(已有) + Phase2-Pan(已有) + PlanB-Pan + P1-C2 + P1-H2 + P1-H4 + P1-F4 = 7个"
echo "GPU1: Exp-E(已有) + Phase2-Foot(已有) + PlanB-Foot + P1-C4 + P1-H3 + P1-F3 + P1-P4 = 7个"
echo ""
echo "预计运行时间: 3-6小时"
echo "监控: grep Iteration logs/*.log | tail -1"
