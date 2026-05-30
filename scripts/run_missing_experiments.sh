#!/bin/bash
# Run missing experiments in parallel:
# GPU0: Pancreas 3v More-densify (long run, ~20 min)
# GPU1: Pancreas 3v Multi-seed stability (6 runs, ~2h total but sequential on GPU1)
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

# ==================== GPU0: Pancreas More-densify ====================
# Uses data/234/ split (consistent with existing chest_densify_test)
MORE_DENSIFY_FLAGS="--densification_interval 50 --densify_grad_threshold 0.0001 \
  --densify_until_iter 20000 --max_num_gaussians 800000"

echo "[GPU0] Starting Pancreas 3v More-densify..."
CUDA_VISIBLE_DEVICES=0 $CONDA train.py \
  -s data/234/pancreas_50_3views.pickle \
  -m output/pancreas_3views_densify \
  --ply_path data/234-sps/init_pancreas_50_3views.npy \
  $COMMON $MORE_DENSIFY_FLAGS > logs/pancreas_densify.log 2>&1 &
PID_DENSIFY=$!
echo "[GPU0] More-densify PID=$PID_DENSIFY"

# ==================== GPU1: Pancreas Stability (6 runs) ====================
ORGAN="pancreas"
for seed in 0 1 2; do
    r2dir="${BASE}/output/${ORGAN}_stability_r2_seed${seed}"
    spdir="${BASE}/output/${ORGAN}_stability_spags_seed${seed}"
    
    # R²-Gaussian baseline
    if [ -d "$r2dir/point_cloud/iteration_30000" ]; then
        echo "✅ [GPU1] $ORGAN R² seed${seed} already done"
    else
        echo "[GPU1] Starting $ORGAN R² seed${seed}..."
        CUDA_VISIBLE_DEVICES=1 $CONDA train.py \
          --seed $seed -s data/369/${ORGAN}_50_3views.pickle \
          -m output/${ORGAN}_stability_r2_seed${seed} \
          --ply_path data/369/init_${ORGAN}_50_3views.npy \
          $COMMON
        echo "✅ [GPU1] $ORGAN R² seed${seed} done"
    fi
    
    # SPAGS
    if [ -d "$spdir/point_cloud/iteration_30000" ]; then
        echo "✅ [GPU1] $ORGAN SPAGS seed${seed} already done"
    else
        echo "[GPU1] Starting $ORGAN SPAGS seed${seed}..."
        CUDA_VISIBLE_DEVICES=1 $CONDA train.py \
          --seed $seed -s data/369/${ORGAN}_50_3views.pickle \
          -m output/${ORGAN}_stability_spags_seed${seed} \
          --ply_path data/369-sps/init_${ORGAN}_50_3views.npy \
          $COMMON $SPAGS_FLAGS
        echo "✅ [GPU1] $ORGAN SPAGS seed${seed} done"
    fi
done

echo "=== GPU1 stability all done! ==="

# Wait for densify to complete
echo "Waiting for More-densify (GPU0) to finish..."
wait $PID_DENSIFY
echo "=== All experiments complete! ==="
