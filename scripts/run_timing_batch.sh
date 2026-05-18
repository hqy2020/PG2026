#!/bin/bash
# Parallel test.py batch on both GPUs
CONDA=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026
cd "$BASE"

ORGANS=(chest head abdomen pancreas foot)

echo "=== GPU0: R²-Gaussian 3-view (2026_04_30) ==="
GPU=0
for organ in "${ORGANS[@]}"; do
  dir="output/2026_04_30_${organ}_3views_r2_gaussian"
  if [ -d "$dir/point_cloud/iteration_30000" ] && [ ! -f "$dir/test/iter_30000/timing_render_test.yml" ]; then
    echo "  GPU${GPU}: ${organ} R²"
    CUDA_VISIBLE_DEVICES=$GPU $CONDA test.py -m "$dir" > /dev/null 2>&1 &
    GPU=$((1 - GPU))  # alternate GPUs
  fi
done
wait
echo "  GPU0/1 R² batch done"

echo "=== GPU0: SPAGS 3-view (2026_04_30) ==="
GPU=0
for organ in "${ORGANS[@]}"; do
  dir="output/2026_04_30_${organ}_3views_spags"
  if [ -d "$dir/point_cloud/iteration_30000" ] && [ ! -f "$dir/test/iter_30000/timing_render_test.yml" ]; then
    echo "  GPU${GPU}: ${organ} SPAGS"
    CUDA_VISIBLE_DEVICES=$GPU $CONDA test.py -m "$dir" > /dev/null 2>&1 &
    GPU=$((1 - GPU))
  fi
done
wait
echo "  GPU0/1 SPAGS batch done"

echo ""
echo "=== GPU0/1: 2026_05_01 2/3/4-view R² ==="
GPU=0
for organ in "${ORGANS[@]}"; do
  for nv in 2 3 4; do
    dir="output/2026_05_01_${organ}_${nv}views_r2_gaussian"
    if [ -d "$dir/point_cloud/iteration_30000" ] && [ ! -f "$dir/test/iter_30000/timing_render_test.yml" ]; then
      echo "  GPU${GPU}: ${organ}_${nv}v R²"
      CUDA_VISIBLE_DEVICES=$GPU $CONDA test.py -m "$dir" > /dev/null 2>&1 &
      GPU=$((1 - GPU))
    fi
  done
done
wait
echo "  Done"

echo "=== GPU0/1: 2026_05_01 2/3/4-view SPAGS ==="
GPU=0
for organ in "${ORGANS[@]}"; do
  for nv in 2 3 4; do
    dir="output/2026_05_01_${organ}_${nv}views_spags"
    if [ -d "$dir/point_cloud/iteration_30000" ] && [ ! -f "$dir/test/iter_30000/timing_render_test.yml" ]; then
      echo "  GPU${GPU}: ${organ}_${nv}v SPAGS"
      CUDA_VISIBLE_DEVICES=$GPU $CONDA test.py -m "$dir" > /dev/null 2>&1 &
      GPU=$((1 - GPU))
    fi
  done
done
wait
echo "  Done"

echo ""
echo "All test.py runs complete!"

# Now re-extract all data
echo ""
echo "=== Re-extracting data ==="
$CONDA scripts/extract_all_data.py
$CONDA scripts/fix_data_extraction.py
echo "Data re-extraction complete!"
