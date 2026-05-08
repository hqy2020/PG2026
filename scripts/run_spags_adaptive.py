#!/usr/bin/env python3
"""
快速 SPAGS 优化实验 — GPU 1 专用
测试 ARIS 建议：自适应阈值 + 渐进衰减
"""
import subprocess, os, sys, time
from pathlib import Path

ROOT = Path("/home/qyhu/Documents/r2_ours/PG2026")
os.chdir(str(ROOT))
PYTHON = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA = "data/234"
SPS = "data/234-sps"
DATE = "2026_05_01"
OUT = ROOT / "output"

# ========== 优化配置 (ARIS 建议) ==========
# 核心变化: 启用自适应阈值 + 渐进衰减
OPTIMIZED_ARGS = [
    "--enable_fsgs_proximity",
    "--gar_proximity_threshold", "0.05",
    "--gar_proximity_k", "5",
    "--gar_adaptive_threshold",           # 🆕 启用自适应阈值！
    "--gar_adaptive_method", "percentile",
    "--gar_adaptive_percentile", "85",     # 只密化最稀疏 15%
    "--gar_progressive_decay",             # 🆕 启用渐进衰减！
    "--gar_decay_start_ratio", "0.7",
    "--gar_final_strength", "0.5",
    "--gar_new_per_source", "1",
    "--gar_max_candidates", "2000",
    "--enable_kplanes",
    "--adm_resolution", "64",
    "--adm_feature_dim", "32",
    "--adm_decoder_hidden", "128",
    "--adm_decoder_layers", "3",
    "--kplanes_lr_init", "0.005",
    "--lambda_plane_tv", "0.0005",
    "--adm_warmup_iters", "15000",
    "--adm_max_range", "0.3",
    "--adm_view_adaptive",
    "--adm_zero_mean",
    "--adm_zero_mean_mode", "density_confidence",
]

# 测试器官: 选最有对比意义的
EXPS = [
    # (organ, views, label)
    ("pancreas", 2, "pan_2v_adaptive"),
    ("pancreas", 3, "pan_3v_adaptive"),
    ("pancreas", 4, "pan_4v_adaptive"),
    ("chest", 3, "chest_3v_adaptive"),
    ("chest", 4, "chest_4v_adaptive"),
    ("foot", 2, "foot_2v_adaptive"),
    ("head", 4, "head_4v_adaptive"),
]

def run_exp(organ, views, label, gpu):
    out_dir = OUT / f"{DATE}_{organ}_{views}views_spags_adaptive"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        PYTHON, "train.py",
        "--method", "r2_gaussian",
        "-s", f"{DATA}/{organ}_50_{views}views.pickle",
        "-m", str(out_dir),
        "--iterations", "30000",
        "--test_iterations", "5000", "10000", "15000", "20000", "25000", "30000",
        "--save_iterations", "30000",
        "--ply_path", f"{SPS}/init_{organ}_50_{views}views.npy",
    ] + OPTIMIZED_ARGS
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"▶ [{label}] GPU{gpu} 启动...", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    
    with open(out_dir / "run.log", "w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(proc.stderr)
    
    if proc.returncode == 0:
        # 读取结果
        eval_dir = out_dir / "eval" / "iter_030000"
        eval_files = list(eval_dir.glob("eval2d_*.yml")) if eval_dir.exists() else []
        psnr_str = "?"
        if eval_files:
            with open(eval_files[0]) as f:
                import yaml
                data = yaml.safe_load(f)
                psnr_str = f"{data.get('psnr_2d', '?'):.2f}"
        print(f"✅ [{label}] 完成! PSNR={psnr_str} ({time.time()-start:.0f}s)", flush=True)
    else:
        err = proc.stderr[-200:] if proc.stderr else "?"
        print(f"❌ [{label}] 失败: {err[:80]}", flush=True)

if __name__ == "__main__":
    import yaml, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    
    print(f"🚀 SPAGS 自适应优化实验 (GPU {args.gpu})")
    print(f"实验数: {len(EXPS)}")
    print(f"配置: adaptive_threshold(percentile, p85) + progressive_decay\n")
    
    # 顺序执行
    for organ, views, label in EXPS:
        run_exp(organ, views, label, args.gpu)
    
    print("\n🏁 全部完成!")
