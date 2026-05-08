#!/usr/bin/env python3
"""
批量渲染 90 个对比实验的新视角合成图，并生成 6×5 网格大图。
输出:
  figures/render_2views.png
  figures/render_3views.png
  figures/render_4views.png
"""

import os, sys, subprocess, time, concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))

PYTHON = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA_DIR = "data/234"
OUTPUT_DIR = "output"
FIGURE_DIR = "figures"
COMPARE_DIR = f"{OUTPUT_DIR}/comparison_234"

os.makedirs(FIGURE_DIR, exist_ok=True)

# 90 个实验定义
METHODS = ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]
METHOD_LABELS = {
    "r2_gaussian": "R²-Gaussian",
    "spags": "SPAGS",
    "xgaussian": "X-Gaussian",
    "fsgs": "FSGS",
    "corgs": "CoR-GS",
    "dngaussian": "DN-Gaussian",
}
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ORGAN_LABELS = {"chest": "Chest", "head": "Head", "abdomen": "Abdomen",
                "foot": "Foot", "pancreas": "Pancreas"}
VIEWS = [2, 3, 4]

# 重跑的 SPAGS 要指向 retry 目录
SPAGS_RETRY = {
    ("pancreas", 2): "2026_05_02_pancreas_2views_spags_retry",
    ("pancreas", 3): "2026_05_02_pancreas_3views_spags_retry",
    ("pancreas", 4): "2026_05_02_pancreas_4views_spags_retry",
    ("foot", 4): "2026_05_02_foot_4views_spags_retry",
}

def get_model_dir(method, organ, views):
    if method == "spags" and (organ, views) in SPAGS_RETRY:
        return f"{OUTPUT_DIR}/{SPAGS_RETRY[(organ, views)]}"
    return f"{OUTPUT_DIR}/2026_05_01_{organ}_{views}views_{method}"

def render_experiment(method, organ, views, gpu):
    """渲染单个实验的 test views，返回渲染图路径"""
    model_dir = get_model_dir(method, organ, views)
    test_dir = f"{model_dir}/test/iter_30000/render_test"
    
    # 如果已经渲染过，跳过
    if os.path.exists(f"{test_dir}/00000_pred.png"):
        return test_dir
    
    data_path = f"{DATA_DIR}/{organ}_50_{views}views.pickle"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    result = subprocess.run(
        [PYTHON, "test.py",
         "-s", data_path,
         "-m", model_dir,
         "--iteration", "30000",
         "--skip_recon", "--skip_render_train"],
        env=env, capture_output=True, text=True, timeout=600
    )
    
    if result.returncode != 0:
        print(f"  ❌ {method}/{organ}/{views}v — {result.stderr[-200:]}")
        return None
    
    return test_dir

# ============================================================
# Step 1: 批量渲染所有 90 个实验
# ============================================================
print("=" * 60)
print("Step 1: 渲染 90 个实验的 test views")
print("=" * 60)

experiments = []
for method in METHODS:
    for organ in ORGANS:
        for views in VIEWS:
            experiments.append((method, organ, views))

# 已渲染的跳过
pending = [(m, o, v) for m, o, v in experiments
           if not os.path.exists(f"{get_model_dir(m, o, v)}/test/iter_30000/render_test/00000_pred.png")]

print(f"总计: {len(experiments)}, 需要渲染: {len(pending)}")

if pending:
    gpu_cycle = [0, 1] * (len(pending) // 2 + 1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, (m, o, v) in enumerate(pending):
            gpu = gpu_cycle[i]
            label = f"[{i+1}/{len(pending)}] {m}/{o}/{v}v → GPU {gpu}"
            print(f"  ▶ {label}")
            future = executor.submit(render_experiment, m, o, v, gpu)
            futures[future] = label
        
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"  ✅ {label}")
                else:
                    print(f"  ❌ {label}")
            except Exception as e:
                print(f"  ❌ {label} — {e}")
else:
    print("  ✅ 全部已渲染，跳过")

# ============================================================
# Step 2: 选择代表视图并生成网格大图
# ============================================================
print("\n")
print("=" * 60)
print("Step 2: 生成 6×5 网格大图")
print("=" * 60)

# 每个器官选一个固定的 test view index
# 取中间视角（对 48-46 个 test views 来说更代表性）
VIEW_INDEX = 20  # 取第 20 个 test view

for views in VIEWS:
    print(f"\n--- {views} Views ---")
    
    # 创建 6×5 的网格
    # 每个子图大小: 256×256
    cell_w, cell_h = 256, 256
    # 方法名行高 + 器官名列宽 + padding
    label_h = 40
    label_w = 80
    padding = 2
    
    grid_w = label_w + 5 * (cell_w + padding) + padding
    grid_h = label_h + 6 * (cell_h + padding) + padding
    
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    
    # 绘制列标题（器官名）
    for col, organ in enumerate(ORGANS):
        x = label_w + col * (cell_w + padding) + padding
        # 简单文字用 PIL 的默认字体
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        draw.text((x + cell_w//2 - 20, 5), ORGAN_LABELS[organ], fill=(0, 0, 0))
    
    # 绘制每行（方法）
    for row, method in enumerate(METHODS):
        y = label_h + row * (cell_h + padding) + padding
        
        # 方法名
        draw = ImageDraw.Draw(canvas)
        draw.text((5, y + cell_h//2 - 10), METHOD_LABELS[method], fill=(0, 0, 0))
        
        for col, organ in enumerate(ORGANS):
            x = label_w + col * (cell_w + padding) + padding
            
            model_dir = get_model_dir(method, organ, views)
            pred_path = f"{model_dir}/test/iter_30000/render_test/{VIEW_INDEX:05d}_pred.png"
            gt_path = f"{model_dir}/test/iter_30000/render_test/{VIEW_INDEX:05d}_gt.png"
            
            if os.path.exists(pred_path):
                img = Image.open(pred_path).convert("L")  # 灰度图
                img = img.resize((cell_w, cell_h), Image.LANCZOS)
                # 转回 RGB 以便合成
                img = img.convert("RGB")
                canvas.paste(img, (x, y))
            else:
                # 缺失标记
                draw.rectangle([x, y, x+cell_w, y+cell_h], fill=(200, 200, 200))
                draw.text((x+cell_w//2-15, y+cell_h//2-5), "N/A", fill=(255, 0, 0))
            
            # 边框
            draw.rectangle([x, y, x+cell_w, y+cell_h], outline=(180, 180, 180), width=1)
    
    # 保存
    out_path = f"{FIGURE_DIR}/render_{views}views.png"
    canvas.save(out_path)
    print(f"  ✅ 保存至: {out_path} ({canvas.size})")

print("\n✅ 全部完成！")
print(f"  输出目录: {FIGURE_DIR}/")
