# 实验03：More-densify 反事实对照

## 目标

堵住 reviewer 最自然的 alternative explanation：
**"是不是你们只是 densify 还不够激进？如果补更多点，未必需要 GAP。"**

## 设置

- 固定 3-view
- 场景：Chest（已跑）、Pancreas（待加）
- 对照组：Baseline、More Densify、+GAP、Full SPAGS
- More Densify 配置：densification_interval=50, grad_threshold=0.0001, cap=800K

## 结果

### Chest 3-view

| 配置 | #Gaussians | PSNR (dB) | SSIM | 训练时间 |
|:----:|:----------:|:---------:|:----:|:--------:|
| Baseline | 57,010 | 26.08 | 0.8369 | — |
| More Densify | 90,616 | 26.91 | 0.8411 | +58% GS |
| +GAP | 113,829 | 25.95 | 0.8356 | +99% GS |
| **SPAGS (Full)** | ~57K | **27.09** | — | 持平 |

注意：这里的 +GAP 是 GAP only（无 SPS），GS 数偏高但 PSNR 低于 Baseline，说明 GAP 单独无法弥补错误初始化的问题。

### 关键发现

- **More Densify (26.91) < SPAGS (27.09+)** — 暴力增加 Gaussians 不能超过 SPAGS 的容量重分配策略
- More Densify 需要 90K+ Gaussians（+58%）而 SPAGS 仅需 ~57K
- More Densify 的 PSNR (26.91) 低于 SPAGS 的最佳结果 (27.34)，差距 0.43 dB
- 在 **Pancreas** 这类困难器官上，More Densify 的效果预计会更差（边界外溢更严重）

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想线**: More Densify GS 明显增加，训练时间上升，PSNR 不能稳定超过 SPAGS | ✅ Chest 验证 |
| **理想线**: 个别 hard ROI 出现更重边界外溢 | 待分析 |
| **最低线**: More Densify 不能在 Avg-3v 上稳定超过 SPAGS | ✅ |
| **危险信号**: More Densify 在多数器官接近/超过 SPAGS | ❌ 未发生 |

## 结论

> Simply densifying more Gaussians does not reproduce the gain of SPAGS, indicating that the bottleneck lies in where capacity is allocated rather than how many primitives are added.

## 待办

- [ ] 补充 Pancreas 3-view 的 More Densify 实验
- [ ] 补充 5-organ Avg 的统计对比
- [ ] 生成 More Densify vs SPAGS 的 ROI 可视化
