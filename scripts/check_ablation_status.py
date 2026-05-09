#!/usr/bin/env python3
"""盘点已有消融实验结果，只看纯ablation 8 configs (GR换GAP)"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path
from glob import glob
import yaml

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))
OUT = "output"
VIEWS = [2, 3, 4]
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]

# 8个消融配置（不带GAP，带GAP）
CONFIGS = {
    "r2_gaussian":      {"label": "R²-Gaussian (基线)",  "sps": False, "adm": False, "gap": False},
    "sps_only":         {"label": "+SPS",                "sps": True,  "adm": False, "gap": False},
    "adm_only":         {"label": "+ADM",                "sps": False, "adm": True,  "gap": False},
    "gap_only":         {"label": "+GAP",                "sps": False, "adm": False, "gap": True},
    "sps_adm":          {"label": "SPS+ADM",             "sps": True,  "adm": True,  "gap": False},
    "sps_gap":          {"label": "SPS+GAP",             "sps": True,  "adm": False, "gap": True},
    "adm_gap":          {"label": "ADM+GAP",             "sps": False, "adm": True,  "gap": True},
    "sps_adm_gap":      {"label": "SPS+ADM+GAP (full)",  "sps": True,  "adm": True,  "gap": True},
}

def find_result(name, organ, views):
    """找已有实验结果"""
    for d in sorted(glob(f"{OUT}/????_??_??_{organ}_{views}views_{name}"), reverse=True):
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                try:
                    ed = yaml.safe_load(f)
                    return d, ed.get('psnr_2d', 0)
                except:
                    pass
    # 也检查旧命名惯例
    for d in sorted(glob(f"{OUT}/2026_05_*_{organ}_{views}views_{name}"), reverse=True):
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                try:
                    ed = yaml.safe_load(f)
                    return d, ed.get('psnr_2d', 0)
                except:
                    pass
    # 检查 gap_ 和 pprune 命名
    for prefix in ["gap_th0p015_r2", "sps_adm_pprune_v1", "sps_adm_gap"]:
        d = f"{OUT}/2026_05_03_{organ}_{views}views_{prefix}"
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                try:
                    ed = yaml.safe_load(f)
                    return d, ed.get('psnr_2d', 0)
                except:
                    pass
    return None, None

# 检查 gap_th0p015_r2 (与 sps_adm_gap 相同参数)
GAP_ALIASES = {
    "sps_adm_gap": "gap_th0p015_r2",
    "adm_gap": "adm_gap",  # need to check if we have this
}

print(f"{'Config':<30} {'2v':>7} {'3v':>7} {'4v':>7}")
print("-" * 52)
for cfg_name, cfg in CONFIGS.items():
    view_hits = []
    for v in VIEWS:
        d, p = find_result(cfg_name, "chest", v)  # 用一个器官判断
        view_hits.append("✅" if d else "⬜")
    print(f"{cfg['label']:<30} {view_hits[0]:>7} {view_hits[1]:>7} {view_hits[2]:>7}")

print()
total = len(CONFIGS) * len(VIEWS) * len(ORGANS)
existing = 0
for cfg_name in CONFIGS:
    for v in VIEWS:
        for organ in ORGANS:
            d, p = find_result(cfg_name, organ, v)
            if d:
                existing += 1
print(f"总计: {total} 组 | 已有: {existing} | 需新跑: {total - existing}")
print(f"预计耗时: {(total-existing)/2*75/60:.1f} 小时 (2 GPU × 75min/组)")
