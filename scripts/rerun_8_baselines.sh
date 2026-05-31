#!/bin/bash
# Re-run 8 baseline experiments on Foot 2v and Pancreas 2v
# Methods: xgaussian, fsgs, corgs, dngaussian (fresh runs)
# Uses seed=42 to differentiate from previous runs

cd /home/qyhu/Documents/r2_ours/PG2026
PY="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
D="data/234"
O="output"

run_one() {
    local method=$1 organ=$2 gpu=$3 seed=$4
    local tag="${method}_rerun_v2"
    local out="${O}/2026_05_31_${organ}_2views_${tag}"
    mkdir -p "$out"
    
    CUDA_VISIBLE_DEVICES=$gpu $PY train.py \
        --method "$method" \
        -s "${D}/${organ}_50_2views.pickle" \
        -m "$out" \
        --iterations 30000 \
        --test_iterations 5000 10000 15000 20000 25000 30000 \
        --save_iterations 30000 \
        --seed "$seed" \
        --ply_path "${D}/init_${organ}_50_2views.npy" \
        > "${out}/run.log" 2>&1
    
    if grep -q 'Training complete' "${out}/run.log" 2>/dev/null; then
        best=$(grep -oP 'psnr2d[ :=]+[0-9.]+' "${out}/run.log" | grep -oP '[0-9.]+' | sort -n | tail -1)
        echo "  ✅ $method $organ 2v (seed=$seed): PSNR=$best"
    else
        echo "  ❌ $method $organ 2v (seed=$seed): FAILED"
    fi
}

echo "Re-running 8 baselines (seed=42) at $(date)"
echo ""

# Wave 1: 4 experiments (2 per GPU)
run_one xgaussian foot 0 42 &
PID1=$!
run_one xgaussian pancreas 1 42 &
PID2=$!
run_one fsgs foot 0 42 &
PID3=$!
run_one fsgs pancreas 1 42 &
PID4=$!
for pid in $PID1 $PID2 $PID3 $PID4; do wait $pid 2>/dev/null; done
echo "Wave 1 done at $(date)"

# Wave 2: 4 experiments (2 per GPU)
run_one corgs foot 0 42 &
PID1=$!
run_one corgs pancreas 1 42 &
PID2=$!
run_one dngaussian foot 0 42 &
PID3=$!
run_one dngaussian pancreas 1 42 &
PID4=$!
for pid in $PID1 $PID2 $PID3 $PID4; do wait $pid 2>/dev/null; done
echo "Wave 2 done at $(date)"

echo ""
echo "=== ALL DONE at $(date) ==="
echo ""
for method in xgaussian fsgs corgs dngaussian; do
    for organ in foot pancreas; do
        log="${O}/2026_05_31_${organ}_2views_${method}_rerun_v2/run.log"
        if [ -f "$log" ]; then
            best=$(grep -oP 'psnr2d[ :=]+[0-9.]+' "$log" | grep -oP '[0-9.]+' | sort -n | tail -1)
            final=$(grep -oP 'ITER 30000.*?psnr2d[ :=]+[0-9.]+' "$log" | grep -oP '[0-9.]+' | tail -1)
            echo "$method $organ: best=$best final=$final"
        fi
    done
done
