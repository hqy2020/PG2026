#!/usr/bin/env python3
"""Generate updated X-Field comparison table with latest SPAGS values."""
import json, yaml, os, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    results = json.load(f)

spags = {}
r2 = {}
for k, v in results.items():
    method, organ, nv = k.split('/')
    if method == 'spags':
        spags[(organ, nv)] = v
    elif method == 'r2_gaussian':
        r2[(organ, nv)] = v

organs = ['chest', 'head', 'abdomen', 'pancreas', 'foot']
views = ['2', '3', '4']

xfield = {}
for organ in organs:
    for nv in views:
        d = f"{BASE}/output/xfield/{organ}_{nv}views_xfield"
        ymls = sorted(glob.glob(f"{d}/eval/iter_*/eval2d_render_test.yml"))
        if ymls:
            with open(ymls[-1]) as f:
                data = yaml.safe_load(f)
            xfield[(organ, nv)] = {'psnr_2d': data['psnr_2d'], 'ssim_2d': data['ssim_2d']}

# Print comparison
header = f"{'Organ':10s} {'View':4s} {'R2 PSNR':10s} {'R2 SSIM':10s} {'SPAGS PSNR':10s} {'SPAGS SSIM':10s} {'XF PSNR':10s} {'XF SSIM':10s}"
print(header)
print('-' * 80)
for organ in organs:
    for nv in views:
        r = r2.get((organ, nv), {})
        s = spags.get((organ, nv), {})
        x = xfield.get((organ, nv), {})
        print(f"{organ:10s} {nv}v  {r.get('psnr_2d', 0):>7.2f}   {r.get('ssim_2d', 0):.4f}   {s.get('psnr_2d', 0):>7.2f}   {s.get('ssim_2d', 0):.4f}   {x.get('psnr_2d', 0):>7.2f}   {x.get('ssim_2d', 0):.4f}")

# Also compute averages
for v in views:
    r2_vals = [r2[(o,v)] for o in organs if (o,v) in r2]
    spags_vals = [spags[(o,v)] for o in organs if (o,v) in spags]
    xf_vals = [xfield[(o,v)] for o in organs if (o,v) in xfield]
    r2_avg = sum(x['psnr_2d'] for x in r2_vals) / len(r2_vals)
    spags_avg = sum(x['psnr_2d'] for x in spags_vals) / len(spags_vals)
    xf_avg = sum(x['psnr_2d'] for x in xf_vals) / len(xf_vals)
    print(f"\n{v}-view Avg: R2={r2_avg:.2f}  SPAGS={spags_avg:.2f}  X-Field={xf_avg:.2f}  SPAGS-XF={spags_avg-xf_avg:+.2f}")
