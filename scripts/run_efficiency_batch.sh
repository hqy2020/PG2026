#!/bin/bash
# Batch run test.py on all 5-organ 3-view R² and SPAGS outputs to get timing & eval data
CONDA=/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python
BASE=/home/qyhu/Documents/r2_ours/PG2026
cd "$BASE"

ORGANS=(chest head abdomen pancreas foot)
DATAS=(2026_05_02 2026_05_02 2026_05_02 2026_05_02 2026_05_02)

echo "=== Timing + GS extraction ==="
echo "Method,Organ,PSNR,SSIM,FPS,ms/view,GS" > logs/efficiency_raw.csv

for i in 0 1 2 3 4; do
    organ=${ORGANS[$i]}
    prefix=${DATAS[$i]}
    
    for method in r2_gaussian spags; do
        dir="output/${prefix}_${organ}_3views_${method}"
        if [ ! -d "$dir/point_cloud/iteration_30000" ]; then
            echo "SKIP $dir (no checkpoint)"
            continue
        fi
        
        # Run test.py (this generates eval + timing yml)
        echo "Running test.py on $dir..."
        $CONDA test.py -m "$dir" > /dev/null 2>&1
        
        # Extract results
        yml="$dir/test/iter_30000/eval2d_render_test.yml"
        tyml="$dir/test/iter_30000/timing_render_test.yml"
        
        if [ -f "$yml" ]; then
            psnr=$(grep 'psnr_2d:' "$yml" | head -1 | awk '{print $2}')
            ssim=$(grep 'ssim_2d:' "$yml" | head -1 | awk '{print $2}')
        else
            psnr="N/A"; ssim="N/A"
        fi
        
        if [ -f "$tyml" ]; then
            fps=$(grep 'fps:' "$tyml" | awk '{print $2}')
            ms=$(grep 'avg_render_time_per_view_ms:' "$tyml" | awk '{print $2}')
        else
            fps="N/A"; ms="N/A"
        fi
        
        # Get GS count from pickle
        pkl="$dir/point_cloud/iteration_30000/point_cloud.pickle"
        gs="N/A"
        if [ -f "$pkl" ]; then
            gs=$($CONDA -c "import pickle; d=pickle.load(open('$pkl','rb')); print(len(d.get('xyz',[])))" 2>/dev/null || echo "ERR")
        fi
        
        echo "$method,$organ,$psnr,$ssim,$fps,$ms,$gs" >> logs/efficiency_raw.csv
        echo "  $method $organ: PSNR=$psnr FPS=$fps GS=$gs"
    done
done

echo ""
echo "=== Full Results ==="
cat logs/efficiency_raw.csv
