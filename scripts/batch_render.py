#!/usr/bin/env python3
"""
Batch render all method predictions for figures.
Run: python batch_render.py
"""
import os, sys, glob, subprocess, time

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

# Define all experiments to render
experiments = []

# 1. SPAGS 3 views (5 organs)
spags_3v = {
    "chest": "2026_04_30_chest_3views_spags",
    "head": "2026_04_30_head_3views_spags",
    "abdomen": "2026_04_30_abdomen_3views_spags",
    "foot": "2026_04_30_foot_3views_spags",
    "pancreas": "2026_04_30_pancreas_3views_spags",
}
for org, dname in spags_3v.items():
    experiments.append((org, "3views", "SPAGS", dname, f"data/369/{org}_50_3views.pickle"))

# 2. R2-Gaussian 3 views
r2_3v = {
    "chest": "2026_04_30_chest_3views_r2_gaussian",
    "head": "2026_04_30_head_3views_r2_gaussian",
    "abdomen": "2026_04_30_abdomen_3views_r2_gaussian",
    "foot": "2026_04_30_foot_3views_r2_gaussian",
    "pancreas": "2026_04_30_pancreas_3views_r2_gaussian",
}
for org, dname in r2_3v.items():
    experiments.append((org, "3views", "R2-Gaussian", dname, f"data/369/{org}_50_3views.pickle"))

# 3. Comparison methods 3 views
methods_3v = {}
for mk in ["xgaussian", "fsgs", "corgs", "dngaussian"]:
    for org in ["chest", "head", "abdomen", "foot", "pancreas"]:
        # Find the actual dir
        dirs = glob.glob(f"{BASE}/output/*{org}*3v*{mk}*")
        if dirs:
            methods_3v[(org, mk)] = dirs[0]

for (org, mk), dname in methods_3v.items():
    label = {"xgaussian": "X-Gaussian", "fsgs": "FSGS", "corgs": "CoR-GS", "dngaussian": "DN-Gaussian"}[mk]
    experiments.append((org, "3views", label, os.path.basename(dname), f"data/369/{org}_50_3views.pickle"))

# 4. SPAGS 2 views and 4 views (need for qual comparison across views)
# Just do chest for now as representative
for view, view_label in [("2", "2views"), ("4", "4views")]:
    dirs = glob.glob(f"{BASE}/output/*chest*{view}v*spags*")
    dirs = [d for d in dirs if 'opt_' not in os.path.basename(d) and 'adaptive' not in os.path.basename(d)]
    if dirs:
        experiments.append(("chest", view_label, "SPAGS", os.path.basename(dirs[0]), f"data/369/chest_50_{view_label}.pickle"))
    
    dirs = glob.glob(f"{BASE}/output/*chest*{view}v*r2_gaussian*")
    dirs = [d for d in dirs if 'opt_' not in os.path.basename(d)]
    if dirs:
        experiments.append(("chest", view_label, "R2-Gaussian", os.path.basename(dirs[0]), f"data/369/chest_50_{view_label}.pickle"))

print(f"Total render jobs: {len(experiments)}")
for org, view, method, dname, data_path in experiments:
    model_path = os.path.join(BASE, "output", dname)
    save_path = os.path.join(model_path, "test", "iter_30000")
    
    # Check if already rendered
    render_dir = os.path.join(save_path, "render_test")
    if os.path.isdir(render_dir) and len(os.listdir(render_dir)) > 10:
        print(f"  ✅ {org} {view} {method}: already rendered ({len(os.listdir(render_dir))} files)")
        continue
    
    print(f"  🔄 {org} {view} {method}: rendering...")
    cmd = f"cd {BASE} && python test.py -m {dname} -s {data_path} --iteration 30000 --skip_recon --skip_render_train"
    
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    
    # Check if render_test was created
    if os.path.isdir(render_dir) and len(os.listdir(render_dir)) > 0:
        n_files = len(os.listdir(render_dir))
        print(f"  ✅ {org} {view} {method}: done ({n_files} files, {elapsed:.1f}s)")
    else:
        print(f"  ❌ {org} {view} {method}: FAILED ({elapsed:.1f}s)")
        if result.stderr:
            print(f"     stderr: {result.stderr[-200:]}")
