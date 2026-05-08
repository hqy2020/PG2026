#!/usr/bin/env bash
# ============================================================================ #
# PG2026 — 完整复现脚本 (2/3/4 Views, 6 Methods × 5 Organs = 90 实验)
# ============================================================================ #
# 用法:
#   bash scripts/reproduce_234.sh                    # 使用 GPU 0,1 (默认)
#   bash scripts/reproduce_234.sh --gpus 0 1 2 3     # 自定义 GPU
#   bash scripts/reproduce_234.sh --methods spags     # 只跑特定方法
#   bash scripts/reproduce_234.sh --summarize-only    # 只汇总已有结果
#
# 输出:
#   output/YYYY_MM_DD_{organ}_{views}views_{method}/  — 每个实验的输出
#   results/results_YYYY_MM_DD_234.{json,md}          — 汇总结果
#
# 论文可复现性: 运行此脚本即可复现论文全部 90 组对比实验。
# 硬件要求: 2×RTX A6000 (48GB)，~15 小时
# ============================================================================ #

set -euo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA_DIR="${PROJECT_DIR}/data/234"
SPS_DIR="${PROJECT_DIR}/data/234-sps"
OUTPUT_DIR="${PROJECT_DIR}/output"
RESULTS_DIR="${PROJECT_DIR}/results"
DATE_STR="$(date +%Y_%m_%d)"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# 迭代步数
ITERATIONS=30000
TEST_ITERS=(5000 10000 15000 20000 25000 30000)
SAVE_ITERS=(30000)

# ─── 方法定义 ────────────────────────────────────────────────────────────────
# 每种方法的 CLI 名称和额外参数 (用 @@ 做占位符，调用时替换)
declare -A METHOD_CLI
declare -A METHOD_ARGS

METHOD_CLI[r2_gaussian]="r2_gaussian"
METHOD_CLI[spags]="r2_gaussian"
METHOD_CLI[xgaussian]="xgaussian"
METHOD_CLI[fsgs]="fsgs"
METHOD_CLI[corgs]="corgs"
METHOD_CLI[dngaussian]="dngaussian"

# SPAGS 参数 (论文核心方法)
SPAGS_ARGS="--enable_fsgs_proximity --gar_proximity_threshold 0.05 \
--gar_proximity_k 5 --no_gar_adaptive_threshold \
--no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000 \
--enable_kplanes --adm_resolution 64 --adm_feature_dim 32 \
--adm_decoder_hidden 128 --adm_decoder_layers 3 \
--kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 \
--adm_warmup_iters 15000 --adm_max_range 0.3 \
--adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence"

METHOD_ARGS[r2_gaussian]="--ply_path ${DATA_DIR}/init_@@organ@@_50_@@views@@views.npy"
METHOD_ARGS[spags]="--ply_path ${SPS_DIR}/init_@@organ@@_50_@@views@@views.npy ${SPAGS_ARGS}"
METHOD_ARGS[xgaussian]="--ply_path ${DATA_DIR}/init_@@organ@@_50_@@views@@views.npy"
METHOD_ARGS[fsgs]="--ply_path ${DATA_DIR}/init_@@organ@@_50_@@views@@views.npy"
METHOD_ARGS[corgs]="--ply_path ${DATA_DIR}/init_@@organ@@_50_@@views@@views.npy"
METHOD_ARGS[dngaussian]="--ply_path ${DATA_DIR}/init_@@organ@@_50_@@views@@views.npy"

# 器官和视角
ORGANS=(chest head abdomen foot pancreas)
VIEWS=(2 3 4)

# ─── 函数 ─────────────────────────────────────────────────────────────────────

# 运行单个实验
run_one() {
    local method="$1" organ="$2" views="$3" gpu="$4"

    # 替换占位符
    local ply_arg="${METHOD_ARGS[$method]}"
    ply_arg="${ply_arg//@@organ@@/$organ}"
    ply_arg="${ply_arg//@@views@@/$views}"

    local output_dir="${OUTPUT_DIR}/${DATE_STR}_${organ}_${views}views_${method}"
    local data_path="${DATA_DIR}/${organ}_50_${views}views.pickle"
    local log_file="${output_dir}/run.log"

    # 跳过已完成的实验
    if [ -f "${output_dir}/eval/iter_030000/eval2d_render_test.yml" ]; then
        psnr=$(grep "psnr_2d:" "${output_dir}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        echo "  ⏭️  已存在: ${organ} ${views}v ${method} (PSNR=${psnr})"
        return 0
    fi

    mkdir -p "$output_dir"
    echo "  ▶ [${method}] ${organ} ${views}v → GPU ${gpu}"
    
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PROJECT_DIR}/train.py" \
        --method "${METHOD_CLI[$method]}" \
        -s "$data_path" \
        -m "$output_dir" \
        --iterations "$ITERATIONS" \
        --test_iterations "${TEST_ITERS[@]}" \
        --save_iterations "${SAVE_ITERS[@]}" \
        $ply_arg \
        > "$log_file" 2>&1
    
    # 验证
    if [ -f "${output_dir}/eval/iter_030000/eval2d_render_test.yml" ]; then
        psnr=$(grep "psnr_2d:" "${output_dir}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        echo "    ✅ ${organ} ${views}v ${method} → PSNR=${psnr}"
    else
        echo "    ❌ ${organ} ${views}v ${method} — 查看日志: $log_file"
    fi
}

# 汇总结果 (扫描所有 output 目录，不限日期)
summarize() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  汇总结果 (扫描全部已有实验)"
    echo "═══════════════════════════════════════════════════════════════════════"
    
    # 读取最完整的汇总 JSON（包含最多实验的那个）
    local best_json=""
    local best_count=0
    for j in "${RESULTS_DIR}"/results_*_234.json; do
        [ -f "$j" ] || continue
        cnt=$(python3 -c "import json; d=json.load(open('$j')); print(len(d))" 2>/dev/null || echo 0)
        [ "$cnt" -gt "$best_count" ] && best_count=$cnt && best_json=$j
    done
    
    if [ -n "$best_json" ]; then
        echo "  使用汇总: $best_json ($best_count 个实验)"
        cat "$best_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
methods = ['r2_gaussian','spags','xgaussian','fsgs','corgs','dngaussian']
organs = ['chest','head','abdomen','foot','pancreas']
views = ['2','3','4']
labels = {'r2_gaussian':'R²-Gaussian','spags':'SPAGS','xgaussian':'X-Gaussian',
          'fsgs':'FSGS','corgs':'CoR-GS','dngaussian':'DN-Gaussian'}
print(f'{\"Method\":<16} {\"2v\":>8} {\"3v\":>8} {\"4v\":>8} {\"Overall\":>8}')
print('-'*52)
for m in methods:
    vals = []
    for v in views:
        vv = [data.get(f'{m}/{o}/{v}',{}).get('psnr_2d',0) for o in organs if data.get(f'{m}/{o}/{v}',{}).get('psnr_2d')]
        avg = sum(vv)/len(vv) if vv else 0
        vals.extend(vv)
        if v == '2': v2=avg
        elif v == '3': v3=avg
        else: v4=avg
    overall = sum(vals)/len(vals) if vals else 0
    n = len(vals)
    print(f'{labels[m]:<16} {v2:>8.2f} {v3:>8.2f} {v4:>8.2f} {overall:>8.2f} (n={n})')
"
    else
        echo "  ⚠️  未找到汇总结果文件。请先运行实验。"
    fi
}

# ─── 主流程 ───────────────────────────────────────────────────────────────────

# 解析参数
GPUS=(0 1)
METHODS_LIST=(r2_gaussian spags xgaussian fsgs corgs dngaussian)
SUMMARIZE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) shift; GPUS=("$@"); break ;;
        --methods) shift; METHODS_LIST=("$@"); break ;;
        --summarize-only) SUMMARIZE_ONLY=true; break ;;
        *) echo "用法: $0 [--gpus N N ...] [--methods m1 m2 ...] [--summarize-only]"; exit 1 ;;
    esac
done

if $SUMMARIZE_ONLY; then
    summarize
    exit 0
fi

# 打印配置
echo "═══════════════════════════════════════════════════════════════════════"
echo "  PG2026 实验复现脚本 — 2/3/4 Views × 6 Methods × 5 Organs"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  日期:    $(date)"
echo "  GPU:     ${GPUS[*]}"
echo "  方法:    ${METHODS_LIST[*]}"
echo "  器官:    ${ORGANS[*]}"
echo "  视角:    ${VIEWS[*]}"
echo "  数据:    ${DATA_DIR}"
echo "  输出:    ${OUTPUT_DIR}/${DATE_STR}_*"
echo "═══════════════════════════════════════════════════════════════════════"

# 生成实验队列
EXPERIMENTS=()
for method in "${METHODS_LIST[@]}"; do
    for organ in "${ORGANS[@]}"; do
        for views in "${VIEWS[@]}"; do
            EXPERIMENTS+=("${method}|${organ}|${views}")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo ""
echo "总计 ${TOTAL} 个实验，GPU 轮询调度..."

# 用 GPU 轮询并行运行
GPU_CYCLE=()
for gpu in "${GPUS[@]}"; do
    for _ in $(seq 1 $(( (TOTAL / ${#GPUS[@]}) + 1 ))); do
        GPU_CYCLE+=("$gpu")
    done
done

# 串行提交（每个 GPU 同时只跑一个，通过 wait 控制）
# 实际并行度 = GPU 数量
RUNNING=0
MAX_JOBS=${#GPUS[@]}
IDX=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r method organ views <<< "$exp"
    gpu="${GPU_CYCLE[$IDX]}"
    
    # 控制并行度
    while [[ $RUNNING -ge $MAX_JOBS ]]; do
        wait -n
        RUNNING=$((RUNNING - 1))
    done
    
    run_one "$method" "$organ" "$views" "$gpu" &
    RUNNING=$((RUNNING + 1))
    IDX=$((IDX + 1))
done

# 等待所有完成
wait
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ✅ 全部 ${TOTAL} 个实验完成!"
echo "═══════════════════════════════════════════════════════════════════════"

# 汇总
summarize

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  结果文件:"
echo "    JSON:    ${RESULTS_DIR}/results_${DATE_STR}_234.json"
echo "    Markdown: ${RESULTS_DIR}/results_${DATE_STR}_234.md"
echo "═══════════════════════════════════════════════════════════════════════"
