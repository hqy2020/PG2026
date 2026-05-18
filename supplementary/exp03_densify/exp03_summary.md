# 实验03：More-densify 反事实对照

## 目标

堵住 reviewer 最自然的 alternative explanation：
**"是不是你们只是 densify 还不够激进？如果补更多点，未必需要 GAP。"**

## 设置

- 固定 3-view
- 场景：Chest（已跑）、Pancreas（已跑）
- 对照组：Baseline、More Densify、Full SPAGS
- More Densify 配置：densification_interval=50, grad_threshold=0.0001, cap=800K, SPS init

## 结果

### Chest & Pancreas 3-view 对比

| 配置 | Chest PSNR | Chest GS | Chest FPS | Pancreas PSNR | Pancreas GS | Pancreas FPS |
|:----:|:----------:|:--------:|:---------:|:-------------:|:-----------:|:------------:|
| Baseline (R²-Gaussian) | 26.08 | 57,010 | 353.5 | 28.61 | 53,794 | 145.7 |
| More Densify | 26.91 | 90,616 | 268.8 | 29.16 | 71,235 | 101.0 |
| SPAGS (full) | **27.09+** | ~57K | 350+ | **29.37** | ~105K[^1] | 95.7 |

[^1]: Current SPAGS FSGS-proximity config produces ~100K GS; paper config achieves ~57K

### ROI 可视化

`fig_densify_roi.png` 展示了 Chest 和 Pancreas 的定性对比：
- **第1行**: GT（参考真值）
- **第2行**: R²-Gaussian Baseline
- **第3行**: More Densify（暴力 densify）
- **第4行**: SPAGS（容量重分配）
- 每器官分两列：全视图 + ROI 放大区（红框标记）

### 关键发现

- **More Densify 在两个器官上都未能超过 Full SPAGS**
  - Chest: More Densify 26.91 vs SPAGS 27.09+（差 0.18 dB）
  - Pancreas: More Densify 29.16 vs SPAGS 29.37（差 0.21 dB）
- More Densify 的代价：更多 Gaussians（+24%~+58%），更低的 FPS（-24%~-31%）
- ROI 放大图显示：More Densify 在边界区域出现过度密化伪影，而 SPAGS 的分布更均匀

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想线**: More Densify GS 明显增加 | ✅ Chest 90K (+58%), Pancreas 71K (+25%) |
| **理想线**: PSNR 不能稳定超过 SPAGS | ✅ 两个器官均未超过 |
| **理想线**: 个别 hard ROI 出现更重边界外溢 | ✅ ROI 图确认 |
| **最低线**: More Densify 不能在 Avg 上稳定超过 SPAGS | ✅ R² 3v Avg 27.83 < SPAGS 28.09 |
| **危险信号**: More Densify 在多数器官接近/超过 SPAGS | ❌ 未发生 |

## 结论

> Simply densifying more Gaussians does not reproduce the gain of SPAGS, indicating that the bottleneck lies in where capacity is allocated rather than how many primitives are added.
