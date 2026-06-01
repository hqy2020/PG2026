import csv
from collections import defaultdict

with open('data_visualization/comparison_all_105.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total experiments: {len(rows)}')
methods = sorted(set(r['method'] for r in rows))
print(f'Methods ({len(methods)}): {methods}')
organs = sorted(set(r['organ'] for r in rows))
print(f'Organs: {organs}')
views = sorted(set(r['views'] for r in rows))
print(f'Views: {views}')
print()

# Show XRAGS rows
print('=== XRAGS entries ===')
for r in rows:
    if r['method'] == 'xrags':
        print(f"  {r['organ']:>10} {r['views']}v  PSNR={r['psnr']}  SSIM={r['ssim']}")
print()

# Compare XRAGS vs R2-Gaussian averages
xrags_vals = {}
r2_vals = {}
for r in rows:
    key = (r['organ'], r['views'])
    if r['method'] == 'xrags':
        xrags_vals[key] = float(r['psnr'])
    elif r['method'] == 'r2_gaussian':
        r2_vals[key] = float(r['psnr'])

print('=== XRAGS vs R²-Gaussian ===')
wins = 0
total = 0
for key in sorted(xrags_vals):
    x = xrags_vals[key]
    r2 = r2_vals.get(key, 0)
    diff = x - r2
    total += 1
    if diff > 0:
        wins += 1
    print(f"  {key[0]:>10} {key[1]}v  XRAGS={x:.4f}  R2={r2:.4f}  Δ={diff:+.4f}")
print(f"Wins: {wins}/{total}")
avg_xrags = sum(xrags_vals.values()) / len(xrags_vals)
avg_r2 = sum(r2_vals.values()) / len(r2_vals)
print(f"Avg XRAGS: {avg_xrags:.4f}, Avg R2: {avg_r2:.4f}, Δ: {avg_xrags-avg_r2:+.4f}")
