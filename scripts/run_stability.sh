#!/bin/bash
# P1-1 Stability experiments: 12 runs × 2 GPUs
# Usage: bash scripts/run_stability.sh [chest|pancreas]
set -e

CONDA=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026
cd "$BASE"

SPAGS_FLAGS="--enable_gap --enable_kplanes \
  --adm_resolution 64 --adm_feature_dim 32 --adm_decoder_hidden 128 \
  --adm_decoder_layers 3 --adm_max_range 0.3 --adm_warmup_iters 15000 \
  --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005"

COMMON="--iterations 30000 --test_iterations 10000 20000 30000"

# Default: chest + pancreas
ORGANS=${1:-"chest pancreas"}

TASKS=()
for organ in $ORGANS; do
    for seed in 0 1 2; do
        r2dir="${BASE}/output/${organ}_stability_r2_seed${seed}"
        spdir="${BASE}/output/${organ}_stability_spags_seed${seed}"
        if [ -d "$r2dir/point_cloud/iteration_30000" ]; then
            echo "✅ r2 ${organ} seed${seed} already done"
        else
            TASKS+=("$organ r2 $seed")
        fi
        if [ -d "$spdir/point_cloud/iteration_30000" ]; then
            echo "✅ spags ${organ} seed${seed} already done"
        else
            TASKS+=("$organ spags $seed")
        fi
    done
done

echo "Remaining: ${#TASKS[@]} tasks"
for ((i=0; i<${#TASKS[@]}; i+=2)); do
    echo "=== Batch $((i/2 + 1)) ==="
    
    read organ method seed <<< "${TASKS[$i]}"
    if [ "$method" = "r2" ]; then
        CMD="CUDA_VISIBLE_DEVICES=0 $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_r2_seed${seed} --ply_path data/369/init_${organ}_50_3views.npy $COMMON"
    else
        CMD="CUDA_VISIBLE_DEVICES=0 $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_spags_seed${seed} --ply_path data/369-sps/init_${organ}_50_3views.npy $COMMON $SPAGS_FLAGS"
    fi
    
    if [ $((i+1)) -lt ${#TASKS[@]} ]; then
        read organ2 method2 seed2 <<< "${TASKS[$((i+1))]}"
        if [ "$method2" = "r2" ]; then
            CMD2="CUDA_VISIBLE_DEVICES=1 $CONDA train.py --seed $seed2 -s data/369/${organ2}_50_3views.pickle -m output/${organ2}_stability_r2_seed${seed2} --ply_path data/369/init_${organ2}_50_3views.npy $COMMON"
        else
            CMD2="CUDA_VISIBLE_DEVICES=1 $CONDA train.py --seed $seed2 -s data/369/${organ2}_50_3views.pickle -m output/${organ2}_stability_spags_seed${seed2} --ply_path data/369-sps/init_${organ2}_50_3views.npy $COMMON $SPAGS_FLAGS"
        fi
        echo "GPU0: $CMD" && echo "GPU1: $CMD2"
        eval "$CMD" & PID0=$!
        eval "$CMD2" & PID1=$!
        wait $PID0 $PID1
    else
        echo "GPU0: $CMD"
        eval "$CMD"
    fi
    echo "Batch $((i/2 + 1)) done"
done
echo "All stability experiments complete!"
