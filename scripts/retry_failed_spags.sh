#!/bin/bash
# 重跑 SPAGS 失败的 4 组实验
# 用法: ./scripts/retry_failed_spags.sh

set -e

BASE="/home/qyhu/Documents/r2_ours/PG2026"
PYTHON="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
cd "$BASE"

DATE=$(date +%Y_%m_%d)
LOG="scripts/retry_failed_${DATE}.log"

SPAGS_ARGS="--method r2_gaussian \
    --iterations 30000 \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --enable_fsgs_proximity --gar_proximity_threshold 0.05 \
    --gar_proximity_k 5 --no_gar_adaptive_threshold \
    --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000 \
    --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
    --adm_decoder_hidden 128 --adm_decoder_layers 3 \
    --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
    --adm_warmup_iters 15000 --adm_max_range 0.3 \
    --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence"

echo "🚀 重跑 SPAGS 失败实验" | tee -a "$LOG"
echo "日期: $(date)" | tee -a "$LOG"

run_experiment() {
    local organ=$1 views=$2 gpu=$3
    local name="${organ}_${views}views_spags_retry"
    local data="data/234/${organ}_50_${views}views.pickle"
    local output="output/${DATE}_${name}"
    local ply="data/234-sps/init_${organ}_50_${views}views.npy"
    
    echo ""
    echo "▶ [$name] GPU${gpu} 启动..." | tee -a "$LOG"
    
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py \
        $SPAGS_ARGS \
        -s "$data" \
        -m "$output" \
        --ply_path "$ply" \
        2>&1 | tee -a "$LOG"
    
    # 检查是否成功
    if [ -f "$output/point_cloud/iteration_30000/point_cloud.pickle" ]; then
        echo "✅ [$name] 完成!" | tee -a "$LOG"
    else
        echo "❌ [$name] 失败" | tee -a "$LOG"
    fi
}

# 按 GPU 并行 (一次跑 2 组)
echo ""
echo "=== Batch 1: Foot 4v (GPU0) + Pancreas 2v (GPU1) ===" | tee -a "$LOG"
run_experiment "foot" 4 0 &
run_experiment "pancreas" 2 1 &
wait

echo ""
echo "=== Batch 2: Pancreas 3v (GPU0) + Pancreas 4v (GPU1) ===" | tee -a "$LOG"
run_experiment "pancreas" 3 0 &
run_experiment "pancreas" 4 1 &
wait

echo ""
echo "🏁 全部完成!" | tee -a "$LOG"
