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

### Chest 3-view (data/234/)

| 配置 | #Gaussians | PSNR (dB) | SSIM | FPS |
|:----:|:----------:|:---------:|:----:|:---:|
| Baseline (R²-Gaussian) | 57,010 | 26.08 | 0.837 | 353.5 |
| More Densify | 90,616 | 26.91 | 0.841 | 268.8 |
| SPAGS (full) | ~57K | 27.09+ | — | 350+ |

### Pancreas 3-view (data/234/)

| 配置 | #Gaussians | PSNR (dB) | SSIM | FPS |
|:----:|:----------:|:---------:|:----:|:---:|
| Baseline (R²-Gaussian) | ~57K | 28.61 | 0.920 | 130+ |
| More Densify | 71,000 | 29.16 | 0.922 | 101.0 |
| SPAGS (full) | ~57K | **29.37** | **0.925** | 130+ |

### 关键发现

- **More Densify 在两个器官上都未能超过 Full SPAGS**
  - Chest: More Densify 26.91 vs SPAGS 27.09+ (差 0.18 dB)
  - Pancreas: More Densify 29.16 vs SPAGS 29.37 (差 0.21 dB)
- More Densify 的代价：更多 Gaussians (+24%~+58%)，更低的 FPS (-24%~-22%)
- 在 Pancreas 这类困难器官上，暴力 densify 带来的收益有限（+0.55 dB），而 SPAGS 的容量重分配策略更高效（+0.76 dB）

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想线**: More Densify GS 明显增加 | ✅ Chest 90K (+58%), Pancreas 71K (+25%) |
| **理想线**: PSNR 不能稳定超过 SPAGS | ✅ 两个器官均未超过 |
| **最低线**: More Densify 不能在 Avg 上稳定超过 SPAGS | ✅ |

## 结论

> Simply densifying more Gaussians does not reproduce the gain of SPAGS, indicating that the bottleneck lies in where capacity is allocated rather than how many primitives are added.
