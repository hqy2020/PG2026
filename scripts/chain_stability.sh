#!/bin/bash
# Chain remaining stability experiments
# Check which runs are done and start pending ones

CONDA=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026

SPAGS_FLAGS="--enable_gap --enable_kplanes \
  --adm_resolution 64 --adm_feature_dim 32 --adm_decoder_hidden 128 \
  --adm_decoder_layers 3 --adm_max_range 0.3 --adm_warmup_iters 15000 \
  --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence \
  --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005"

COMMON="--iterations 30000 --test_iterations 10000 20000 30000"

cd "$BASE"

# Define all runs
declare -A RUNS
declare -A GPU
declare -A STATUS

RUNS[0]="pancreas r2 0"
RUNS[1]="pancreas spags 0"
RUNS[2]="pancreas r2 1"
RUNS[3]="pancreas spags 1"
RUNS[4]="pancreas r2 2"
RUNS[5]="pancreas spags 2"

# Check which are done
for i in "${!RUNS[@]}"; do
    read organ method seed <<< "${RUNS[$i]}"
    dir="${BASE}/output/${organ}_stability_${method}_seed${seed}"
    if [ -d "$dir/point_cloud/iteration_30000" ]; then
        STATUS[$i]="DONE"
    elif pgrep -f "stability_${method}_seed${seed}" > /dev/null 2>&1; then
        STATUS[$i]="RUNNING"
    else
        STATUS[$i]="PENDING"
    fi
done

echo "=== Stability Run Status ==="
for i in "${!RUNS[@]}"; do
    read organ method seed <<< "${RUNS[$i]}"
    echo "  [$i] ${organ}_${method}_seed${seed}: ${STATUS[$i]}"
done

# Find a free GPU
for gpu in 0 1; do
    if ! pgrep -f "CUDA_VISIBLE_DEVICES=$gpu" > /dev/null 2>&1; then
        if nvidia-smi -i $gpu --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -q "^ 0 "; then
            FREE_GPU=$gpu
            break
        fi
    fi
done

# Start next pending run on free GPU
for i in "${!RUNS[@]}"; do
    if [ "${STATUS[$i]}" = "PENDING" ]; then
        read organ method seed <<< "${RUNS[$i]}"
        echo "Starting [$i] ${organ}_${method}_seed${seed} on GPU:${FREE_GPU:-0}..."
        
        if [ "$method" = "r2" ]; then
            CMD="CUDA_VISIBLE_DEVICES=${FREE_GPU:-0} $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_${method}_seed${seed} --ply_path data/369/init_${organ}_50_3views.npy $COMMON"
        else
            CMD="CUDA_VISIBLE_DEVICES=${FREE_GPU:-0} $CONDA train.py --seed $seed -s data/369/${organ}_50_3views.pickle -m output/${organ}_stability_${method}_seed${seed} --ply_path data/369-sps/init_${organ}_50_3views.npy $COMMON $SPAGS_FLAGS"
        fi
        
        echo "$CMD"
        eval "$CMD" 2>&1 | tee logs/${organ}_stability_${method}_seed${seed}.log
        echo "Done: [$i] ${organ}_${method}_seed${seed}"
        exit 0
    fi
done

echo "All stability runs complete!"
