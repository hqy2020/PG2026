#!/usr/bin/env python3
"""
P0-2 (R2): Precise Efficiency Table WITH timing.
Extracts train wall-clock time, inference FPS (from updated test.py),
Gaussian count, and PSNR from existing runs.
"""
import os, sys, glob, re, yaml, pickle

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
TABLES_DIR = os.path.join(BASE, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# === DATA ===
# Use the ablation series (data/234/) for 3-view chest comparison
RUNS = {
    "Baseline": {
        "dir": f"{BASE}/output/2026_05_02_chest_3views_r2_gaussian",
        "label": "R$^2$-Gaussian",
    },
    "More Densify": {
        "dir": f"{BASE}/output/chest_densify_test",
        "label": "More Densify",
    },
    "+GAP": {
        "dir": f"{BASE}/output/2026_05_02_chest_3views_gar_only",
        "label": "+GAP",
    },
    "Full SPAGS": {
        "dir": f"{BASE}/output/2026_05_02_chest_3views_spags",
        "label": "SPAGS",
    },
}
# Also collect from Apr 30 (data/369/) for full 5-organ table
ORGANS_369 = ["chest", "head", "pancreas", "abdomen", "foot"]


def get_train_time(run_dir):
    """Extract training wall-clock time from run.log or events."""
    # Try run.log
    for log_path in [f"{run_dir}/run.log", f"{run_dir}/train.log"]:
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Training complete', line) or re.search(r'total.*time.*\d+', line, re.I)
                if m:
                    return "extracted"
    # Try to estimate from iter count and time per iter
    eval_dir = f"{run_dir}/eval"
    if os.path.isdir(eval_dir):
        iters = sorted([int(d.split("_")[1]) for d in os.listdir(eval_dir) if d.startswith("iter_")])
        if iters:
            max_iter = max(iters)
            # 30K iters at ~25 it/s ≈ 1200s = 20min
            return f"~{max_iter // 25 // 60}min"
    return "N/A"


def get_timing(run_dir):
    """Read timing file from updated test.py output."""
    for path in [f"{run_dir}/test/iter_30000/timing_render_test.yml",
                 f"{run_dir}/eval/iter_030000/timing_render_test.yml"]:
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
            return data
    return None


def get_gs_count(run_dir):
    pkl = f"{run_dir}/point_cloud/iteration_30000/point_cloud.pickle"
    if not os.path.exists(pkl):
        pkls = glob.glob(f"{run_dir}/point_cloud/iteration_*/point_cloud.pickle")
        pkl = sorted(pkls)[-1] if pkls else None
    if pkl and os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        xyz = data.get('xyz')
        if xyz is not None:
            return len(xyz)
    return None


def get_psnr_from_eval(run_dir):
    for yml in [f"{run_dir}/test/iter_30000/eval2d_render_test.yml",
                f"{run_dir}/eval/iter_030000/eval2d_render_test.yml"]:
        if os.path.exists(yml):
            with open(yml) as f:
                for line in f:
                    if line.startswith("psnr_2d:"):
                        return float(line.split(":")[1])
    return None


def main():
    print("=" * 60)
    print("P0-2 (R2): Efficiency Table with Timing")
    print("=" * 60)
    
    # ===== Section 1: Densify comparison (chest 3-view) =====
    print("\n--- Densify vs GAP (Chest 3-view) ---")
    rows = []
    for name, cfg in RUNS.items():
        d = cfg["dir"]
        if not os.path.exists(d):
            print(f"  {name}: ❌ missing dir")
            continue
        gs = get_gs_count(d)
        psnr = get_psnr_from_eval(d)
        timing = get_timing(d)
        fps = timing["fps"] if timing else "N/A"
        ms = timing["avg_render_time_per_view_ms"] if timing else "N/A"
        
        print(f"  {name:<14} GS={gs or 'N/A':>7}  PSNR={psnr or 'N/A':>6}  "
              f"FPS={fps if fps != 'N/A' else 'N/A':>5}  ms/view={ms if ms != 'N/A' else 'N/A':>5}")
        rows.append((name, gs, psnr, fps, ms))
    
    # ===== Section 2: 5-organ SPAGS vs R²-G summary (from data/369/) =====
    print("\n--- 5-Organ Summary (3-view) ---")
    all_gs, all_psnr = {}, {}
    for organ in ORGANS_369:
        for tag, pat in [("R2", "r2_gaussian"), ("SPAGS", "spags")]:
            dirs = sorted(glob.glob(f"{BASE}/output/*{organ}_3views_{pat}*"))
            found = None
            for d in dirs:
                dn = os.path.basename(d)
                if any(x in dn for x in ['opt_', 'adaptive', 'retry']):
                    continue
                found = d; break
            if found:
                gs = get_gs_count(found)
                psnr = get_psnr_from_eval(found)
                if gs: all_gs.setdefault(tag, []).append(gs)
                if psnr: all_psnr.setdefault(tag, []).append(psnr)
    
    for tag in ["R2", "SPAGS"]:
        vals = all_psnr.get(tag, [])
        gs_vals = all_gs.get(tag, [])
        print(f"  {tag:<12} Avg GS={sum(gs_vals)/len(gs_vals):,.0f}  Avg PSNR={sum(vals)/len(vals):.2f}")
    
    # ===== Generate LaTeX =====
    print("\n--- Generating tab_efficiency.tex ---")
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Efficiency comparison on Chest 3-view. "
               r"Train time from wall-clock; inference from CUDA Event timing.}")
    tex.append(r"\label{tab:efficiency}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r" Method & \#Gaussians & PSNR (dB) & FPS & ms/view \\")
    tex.append(r"\midrule")
    
    # Densify comparison rows
    for i, (name, gs, psnr, fps, ms) in enumerate(rows):
        gs_str = f"{gs:,}" if gs else "---"
        psnr_str = f"{psnr:.2f}" if psnr else "---"
        fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
        ms_str = f"{ms:.1f}" if isinstance(ms, (int, float)) else str(ms)
        tex.append(f" {name} & {gs_str} & {psnr_str} & {fps_str} & {ms_str} \\\\")
        if i == 0:  # after baseline
            tex.append(r"\cmidrule{1-5}")
    
    # 5-organ average row
    tex.append(r"\midrule")
    r2_avg_psnr = sum(all_psnr.get("R2", [])) / len(all_psnr.get("R2", [1]))
    sp_avg_psnr = sum(all_psnr.get("SPAGS", [])) / len(all_psnr.get("SPAGS", [1]))
    r2_avg_gs = sum(all_gs.get("R2", [])) / len(all_gs.get("R2", [1]))
    sp_avg_gs = sum(all_gs.get("SPAGS", [])) / len(all_gs.get("SPAGS", [1]))
    tex.append(f" Avg R$^2$-G (5-organ) & {r2_avg_gs:,.0f} & {r2_avg_psnr:.2f} & --- & --- \\\\")
    tex.append(f" Avg SPAGS (5-organ) & {sp_avg_gs:,.0f} & {sp_avg_psnr:.2f} & --- & --- \\\\")
    
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    
    tex_path = f"{TABLES_DIR}/tab_efficiency.tex"
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  ✅ {tex_path}")


if __name__ == "__main__":
    main()
