#!/bin/bash
# Run 2 per GPU in parallel, wave by wave

cd /home/qyhu/Documents/r2_ours/PG2026
PY="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
D="data/234"
O="output"

run_one() {
    local method=$1 organ=$2 seed=$3 gpu=$4
    local out="${O}/2026_05_31_${organ}_2views_${method}_seed${seed}"
    mkdir -p "$out"
    
    # Skip if already completed
    if [ -f "${out}/eval/iter_030000/eval2d_render_test.yml" ]; then
        return 0
    fi
    
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
    
    if [ -f "${out}/eval/iter_030000/eval2d_render_test.yml" ]; then
        echo "  ✅ $method $organ 2v seed=$seed (GPU$gpu): DONE"
    else
        echo "  ❌ $method $organ 2v seed=$seed (GPU$gpu): FAILED"
    fi
}

# Queue of experiments
queue=()
for method in xgaussian fsgs corgs dngaussian r2_gaussian; do
    for organ in foot pancreas; do
        for seed in 1 2; do
            queue+=("$method|$organ|$seed")
        done
    done
done

echo "Starting ${#queue[@]} baseline reseeds at $(date)"
echo ""

# Process 4 at a time (2 per GPU)
pos=0
while [ $pos -lt ${#queue[@]} ]; do
    pids=""
    for slot in 0 1 2 3; do
        idx=$((pos + slot))
        [ $idx -ge ${#queue[@]} ] && continue
        IFS='|' read -r m o s <<< "${queue[$idx]}"
        gpu=$((slot % 2))
        run_one "$m" "$o" "$s" "$gpu" &
        pids="$pids $!"
    done
    # Wait for this wave
    for pid in $pids; do wait $pid 2>/dev/null; done
    pos=$((pos + 4))
    echo "Wave $((pos/4)) complete ($pos/${#queue[@]}) at $(date)"
done

echo ""
echo "✅ ALL DONE at $(date)"
