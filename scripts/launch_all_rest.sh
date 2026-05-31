#!/bin/bash
# Launch remaining ablation experiments alongside current runs
cd /home/qyhu/Documents/r2_ours/PG2026
PY="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA_CHEST="data/234/chest_50_3views.pickle"

run_bg() {
    local name=$1 gpu=$2 extra_args=$3
    local out="output/${name}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py \
        --method r2_gaussian \
        -s "$DATA_CHEST" \
        -m "$out" \
        --iterations 30000 \
        --test_iterations 5000 10000 15000 20000 25000 30000 \
        --save_iterations 30000 \
        $extra_args \
        > "${out}/run.log" 2>&1 &
    echo "Launched $name on GPU$gpu (PID $!)"
}

echo "Launching ablation experiments at $(date)"
echo ""

# SPS α sweep (B+SPS init only)
# GPU0 gets 2 SPS (alongside 2 XGaussian)
# GPU1 gets 2 SPS, 2 GAP (alongside 2 XGaussian + 1 ADM)

run_bg "2026_05_31_chest_3views_sps_alpha0.0" 0 \
    "--ply_path data/sps-alpha/alpha0.0/init_chest_50_3views.npy"

run_bg "2026_05_31_chest_3views_sps_alpha0.1" 0 \
    "--ply_path data/sps-alpha/alpha0.1/init_chest_50_3views.npy"

run_bg "2026_05_31_chest_3views_sps_alpha0.5" 1 \
    "--ply_path data/sps-alpha/alpha0.5/init_chest_50_3views.npy"

run_bg "2026_05_31_chest_3views_sps_alpha1.0" 1 \
    "--ply_path data/sps-alpha/alpha1.0/init_chest_50_3views.npy"

# GAP τ sweep (B+GAP only)
run_bg "2026_05_31_chest_3views_gap_tau0.005" 1 \
    "--enable_gap --gap_threshold 0.005 --gap_k 5 --gap_max_ratio 0.02 --gap_start_iter 2000 --gap_until_iter 20000 --gap_interval 500 --gap_gradient_aware --gap_gradient_threshold 0.0002"

run_bg "2026_05_31_chest_3views_gap_tau0.030" 1 \
    "--enable_gap --gap_threshold 0.030 --gap_k 5 --gap_max_ratio 0.02 --gap_start_iter 2000 --gap_until_iter 20000 --gap_interval 500 --gap_gradient_aware --gap_gradient_threshold 0.0002"

echo ""
echo "All launched! Waiting for completion..."
echo "Time: $(date)"
echo ""
echo "Current processes:"
ps aux | grep "train.py" | grep -v grep | wc -l
echo ""
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Monitor until all done
while true; do
    n=$(ps aux | grep "train.py" | grep -v grep | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 120
    echo "[$(date)] $n processes remaining"
done

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE at $(date)"
echo "============================================"
