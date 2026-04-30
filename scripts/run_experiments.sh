#!/bin/bash
# PG2026 Experiment Runner
# Usage: ./run_experiments.sh <config> <organ> <views> <gpu>
#   config: baseline|sps|gar|adm|sps_gar|sps_adm|gar_adm|spags
#   organ: foot|chest|head|abdomen|pancreas
#   views: 3|6|9
set -e
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

CONFIG=$1; ORGAN=$2; VIEWS=$3; GPU=${4:-0}
DATA_DIR="data/369"
INIT_DIR="data/369"
OUTPUT="output/2026_04_29_${ORGAN}_${VIEWS}views_${CONFIG}"

# 通用参数
BASE_ARGS="--method r2_gaussian -s ${DATA_DIR}/${ORGAN}_50_${VIEWS}views.pickle"
BASE_ARGS+=" -m ${OUTPUT} --iterations 30000 --test_iterations 10000 20000 30000"

case $CONFIG in
  baseline)
    python train.py $BASE_ARGS --ply_path ${INIT_DIR}/init_${ORGAN}_50_${VIEWS}views.npy
    ;;
  sps)
    python train.py $BASE_ARGS --ply_path ${DATA_DIR}-sps/init_${ORGAN}_50_${VIEWS}views.npy
    ;;
  gar)
    python train.py $BASE_ARGS --ply_path ${INIT_DIR}/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_fsgs_proximity --gar_proximity_threshold 0.05 \
      --gar_proximity_k 5 --no_gar_adaptive_threshold \
      --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000
    ;;
  adm)
    python train.py $BASE_ARGS --ply_path ${INIT_DIR}/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
      --adm_decoder_hidden 128 --adm_decoder_layers 3 \
      --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
      --adm_warmup_iters 15000 --adm_max_range 0.3 \
      --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence
    ;;
  sps_gar)
    python train.py $BASE_ARGS --ply_path ${DATA_DIR}-sps/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_fsgs_proximity --gar_proximity_threshold 0.05 \
      --gar_proximity_k 5 --no_gar_adaptive_threshold \
      --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000
    ;;
  sps_adm)
    python train.py $BASE_ARGS --ply_path ${DATA_DIR}-sps/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
      --adm_decoder_hidden 128 --adm_decoder_layers 3 \
      --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
      --adm_warmup_iters 15000 --adm_max_range 0.3 \
      --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence
    ;;
  gar_adm)
    python train.py $BASE_ARGS --ply_path ${INIT_DIR}/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_fsgs_proximity --gar_proximity_threshold 0.05 \
      --gar_proximity_k 5 --no_gar_adaptive_threshold \
      --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000 \
      --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
      --adm_decoder_hidden 128 --adm_decoder_layers 3 \
      --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
      --adm_warmup_iters 15000 --adm_max_range 0.3 \
      --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence
    ;;
  spags)
    python train.py $BASE_ARGS --ply_path ${DATA_DIR}-sps/init_${ORGAN}_50_${VIEWS}views.npy \
      --enable_fsgs_proximity --gar_proximity_threshold 0.05 \
      --gar_proximity_k 5 --no_gar_adaptive_threshold \
      --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000 \
      --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
      --adm_decoder_hidden 128 --adm_decoder_layers 3 \
      --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
      --adm_warmup_iters 15000 --adm_max_range 0.3 \
      --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence
    ;;
  *)
    echo "Unknown config: $CONFIG. Use: baseline|sps|gar|adm|sps_gar|sps_adm|gar_adm|spags"
    exit 1
    ;;
esac
