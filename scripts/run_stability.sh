#!/bin/bash
# Launch all P1-1 stability experiments sequentially, 2 at a time
# Usage: bash scripts/run_stability.sh
set -e

CONDA=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026
cd "$BASE"

SPAGS_FLAGS="--enable_fsgs_proximity --gar_proximity_threshold 0.05 --gar_proximity_k 5 \
  --no_gar_adaptive_threshold --no_gar_progressive_decay \
  --gar_new_per_source 1 --gar_max_candidates 2000 \
  --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
  --adm_decoder_hidden 128 --adm_decoder_layers 3 \
  --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
  --adm_warmup_iters 15000 --adm_max_range 0.3 \
  --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence"

COMMON="--iterations 30000 --test_iterations 10000 20000 30000"

TASKS=()
for organ in chest pancreas; do
    for method in r2 spags; do
        for seed in 0 1 2; do
            dir="${BASE}/output/${organ}_stability_${method}_seed${seed}"
            if [ -d "$dir/point_cloud/iteration_30000" ]; then
                echo "✅ $dir already done, skip"
                continue
            fi
            TASKS+=("$organ $method $seed")
        done
    done
done

echo "Remaining tasks: ${#TASKS[@]}"
for ((i=0; i<${#TASKS[@]}; i+=2)); do
    echo ""
    echo "=== Batch $((i/2 + 1)) ==="
    
    # Task on GPU 0
    read organ method seed <<< "${TASKS[$i]}"
    if [ "$method" = "r2" ]; then
        CMD="CUDA_VISIBLE_DEVICES=0 $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_r2_seed${seed} --ply_path data/369/init_${organ}_50_3views.npy $COMMON"
    else
        CMD="CUDA_VISIBLE_DEVICES=0 $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_spags_seed${seed} --ply_path data/369-sps/init_${organ}_50_3views.npy $COMMON $SPAGS_FLAGS"
    fi
    echo "GPU0: $CMD"
    
    # Task on GPU 1 (if exists)
    if [ $((i+1)) -lt ${#TASKS[@]} ]; then
        read organ2 method2 seed2 <<< "${TASKS[$((i+1))]}"
        if [ "$method2" = "r2" ]; then
            CMD2="CUDA_VISIBLE_DEVICES=1 $CONDA train.py --seed $seed2 -s data/369/${organ2}_50_3views.pickle -m output/${organ2}_stability_r2_seed${seed2} --ply_path data/369/init_${organ2}_50_3views.npy $COMMON"
        else
            CMD2="CUDA_VISIBLE_DEVICES=1 $CONDA train.py --seed $seed2 -s data/369/${organ2}_50_3views.pickle -m output/${organ2}_stability_spags_seed${seed2} --ply_path data/369-sps/init_${organ2}_50_3views.npy $COMMON $SPAGS_FLAGS"
        fi
        echo "GPU1: $CMD2"
        # Run both in parallel and wait
        eval "$CMD" &
        PID0=$!
        eval "$CMD2" &
        PID1=$!
        wait $PID0 $PID1
        echo "Batch $((i/2 + 1)) complete"
    else
        eval "$CMD"
        echo "Last task complete"
    fi
done
echo ""
echo "All stability experiments complete!"
