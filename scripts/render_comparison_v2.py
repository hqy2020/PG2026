#!/usr/bin/env python3
"""
PG2026 对比可视化渲染器
为所有 6 个方法渲染新视角合成图，并生成 6×5 网格大图（含 GT 行）。
支持各方法的独立 checkpoint 格式。
"""

import os, sys, importlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

PROJECT = Path(os.path.abspath(__file__)).parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

PYTHON = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA_DIR = "data/234"
OUTPUT_DIR = "output"
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

METHODS = ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]
METHOD_LABELS = {"r2_gaussian": "R²-Gaussian", "spags": "SPAGS",
    "xgaussian": "X-Gaussian", "fsgs": "FSGS", "corgs": "CoR-GS", "dngaussian": "DN-Gaussian"}
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ORGAN_LABELS = {"chest": "Chest", "head": "Head", "abdomen": "Abdomen",
                "foot": "Foot", "pancreas": "Pancreas"}
VIEWS = [2, 3, 4]

SPAGS_RETRY = {("pancreas", 2): "2026_05_02_pancreas_2views_spags_retry",
    ("pancreas", 3): "2026_05_02_pancreas_3views_spags_retry",
    ("pancreas", 4): "2026_05_02_pancreas_4views_spags_retry",
    ("foot", 4): "2026_05_02_foot_4views_spags_retry"}

def get_model_dir(method, organ, views):
    if method == "spags" and (organ, views) in SPAGS_RETRY:
        return f"{OUTPUT_DIR}/{SPAGS_RETRY[(organ, views)]}"
    return f"{OUTPUT_DIR}/2026_05_01_{organ}_{views}views_{method}"

def render_method(method, organ, views, device="cuda:0"):
    """渲染指定方法/器官/视角的一个 test view"""
    model_dir = get_model_dir(method, organ, views)
    test_dir = f"{model_dir}/test/iter_30000/render_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # 如果已经渲染过，跳过
    if os.path.exists(f"{test_dir}/00000_pred.png"):
        return True
    
    data_path = f"{DATA_DIR}/{organ}_50_{views}views.pickle"
    
    print(f"  ▶ {method}/{organ}/{views}v", flush=True)
    
    try:
        if method in ["r2_gaussian", "spags", "fsgs", "corgs", "dngaussian"]:
            # 这些方法都有 point_cloud.pickle，用 test.py 渲染
            import subprocess
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")
            r = subprocess.run(
                [PYTHON, "test.py", "-s", data_path, "-m", model_dir,
                 "--iteration", "30000", "--skip_recon", "--skip_render_train"],
                env=env, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print(f"    ⚠️  test.py 返回 {r.returncode}: {r.stderr[-200:]}")
                return False
            return True
            
        elif method == "xgaussian":
            # X-Gaussian 没有 point_cloud.pickle，需独立处理
            torch.cuda.set_device(device)
            
            from r2_gaussian.dataset import Scene
            from r2_gaussian.arguments import ModelParams, PipelineParams
            from r2_gaussian.baselines.xgaussian.model import XGaussianModel
            from r2_gaussian.baselines.xgaussian.renderer import render_xgaussian
            from r2_gaussian.baselines.xgaussian.config import XGaussianConfig
            
            # 解析参数
            import argparse
            parser = argparse.ArgumentParser()
            mp = ModelParams(parser)
            pp = PipelineParams(parser)
            args = parser.parse_args(["-s", data_path, "-m", model_dir])
            
            # 加载场景
            scene = Scene(mp.extract(args), shuffle=False)
            
            # 创建模型（先初始化一个空的）
            gaussians = XGaussianModel(sh_degree=3, scale_bound=None)
            gaussians = gaussians.to(device)
            
            # 加载 checkpoint（只加载权重，不需要 optimizer）
            ckpt_path = f"{model_dir}/xgaussian_iter_30000.pth"
            if not os.path.exists(ckpt_path):
                ckpt_path = f"{model_dir}/chkpnt_xgaussian_30000.pth"
            
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                if isinstance(ckpt, tuple):
                    # capture() 格式: (state_dict, iteration)
                    state = ckpt[0]
                elif isinstance(ckpt, dict):
                    state = ckpt
                else:
                    state = ckpt
                
                # 手动恢复模型参数（跳过 optimizer 设置）
                for key in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
                    if key in state:
                        setattr(gaussians, key, state[key])
                if 'max_radii2D' in state:
                    gaussians.max_radii2D = state['max_radii2D']
                if 'spatial_lr_scale' in state:
                    gaussians.spatial_lr_scale = state['spatial_lr_scale']
            else:
                # Fallback: 从点云初始化一个空的
                print(f"    ⚠️  checkpoint 不存在, 尝试 point_cloud.pickle")
                if os.path.exists(f"{model_dir}/point_cloud/iteration_30000/point_cloud.pickle"):
                    import pickle
                    with open(f"{model_dir}/point_cloud/iteration_30000/point_cloud.pickle", 'rb') as f:
                        pcd = pickle.load(f)
                    gaussians.create_from_pcd(pcd['xyz'], 1.0)
                else:
                    return False
            
            scene.gaussians = gaussians
            gaussians.eval()
            
            # 渲染 test views
            pipe = pp.extract(args)
            test_cameras = scene.getTestCameras()
            
            from torchvision.utils import save_image
            with torch.no_grad():
                for idx, viewpoint in enumerate(test_cameras):
                    render_result = render_xgaussian(viewpoint, gaussians, pipe)
                    image = render_result["render"]
                    gt_image = viewpoint.original_image.to(image.device)
                    save_image(image, f"{test_dir}/{idx:05d}_pred.png")
                    save_image(gt_image[0:1], f"{test_dir}/{idx:05d}_gt.png")
            
            return True
    
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# 渲染所有实验
# ============================================================
print("=" * 60)
print("渲染所有实验的新视角合成图")
print("=" * 60)

total, success = 0, 0
for method in METHODS:
    for organ in ORGANS:
        for views in VIEWS:
            total += 1
            ok = render_method(method, organ, views, "cuda:0")
            if ok:
                success += 1
            status = "✅" if ok else "❌"
            print(f"  {status} [{success}/{total}] {method}/{organ}/{views}v", flush=True)

print(f"\n✅ 渲染完成: {success}/{total}")

# ============================================================
# 生成网格大图（含 GT 行）
# ============================================================
print("\n" + "=" * 60)
print("生成 7×5 网格大图（含 GT 行）")
print("=" * 60)

VIEW_INDEX = 20  # 固定 test view index
CELL_W, CELL_H = 256, 256
LABEL_H = 50
LABEL_W = 100
PAD = 3

for views in VIEWS:
    print(f"\n--- {views} Views ---")
    
    # 7 行: GT + 6 方法; 5 列: 器官
    rows = ["GT"] + METHODS
    row_labels = {"GT": "Ground Truth"}
    row_labels.update(METHOD_LABELS)
    
    grid_w = LABEL_W + 5 * (CELL_W + PAD) + PAD
    grid_h = LABEL_H + 7 * (CELL_H + PAD) + PAD
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # 列标题（器官名）
    for col, organ in enumerate(ORGANS):
        x = LABEL_W + col * (CELL_W + PAD) + PAD
        text = ORGAN_LABELS[organ]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (CELL_W - tw)//2, 8), text, fill=(40, 40, 40), font=font)
    
    # 画每行
    for row_idx, row_name in enumerate(rows):
        y = LABEL_H + row_idx * (CELL_H + PAD) + PAD
        
        # 行标签
        label = row_labels[row_name]
        draw.text((8, y + CELL_H//2 - 10), label, fill=(40, 40, 40), font=small_font)
        
        for col, organ in enumerate(ORGANS):
            x = LABEL_W + col * (CELL_W + PAD) + PAD
            
            if row_name == "GT":
                # Ground Truth: 用 r2_gaussian 的 GT 图（所有方法 GT 相同）
                ref_dir = get_model_dir("r2_gaussian", organ, views)
                img_path = f"{ref_dir}/test/iter_30000/render_test/{VIEW_INDEX:05d}_gt.png"
            else:
                img_path = f"{get_model_dir(row_name, organ, views)}/test/iter_30000/render_test/{VIEW_INDEX:05d}_pred.png"
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L").convert("RGB")
                img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x+CELL_W, y+CELL_H], fill=(220, 220, 220))
                draw.text((x+CELL_W//2-20, y+CELL_H//2-8), "N/A", fill=(200, 50, 50), font=small_font)
            
            # 边框
            draw.rectangle([x, y, x+CELL_W, y+CELL_H], outline=(160, 160, 160), width=1)
        
        # 行分割线
        if row_idx == 0:  # GT 行特殊标记
            draw.line([(LABEL_W, y+CELL_H+PAD//2), (grid_w-PAD, y+CELL_H+PAD//2)], 
                      fill=(100, 100, 100), width=2)
    
    out_path = f"{FIGURE_DIR}/render_{views}views.png"
    canvas.save(out_path)
    print(f"  ✅ 保存至: {out_path} ({canvas.size[0]}×{canvas.size[1]})")

print(f"\n✅ 全部完成!")
print(f"  输出: {FIGURE_DIR}/render_*.png")
