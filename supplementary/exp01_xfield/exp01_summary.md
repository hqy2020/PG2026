# 实验01：X-Field 同协议 2/3/4-view 重跑

## 目标

回答 reviewer 最危险的追问：**"你有没有正面比较 2025 以后最接近的 CT/X-ray direct baseline？"**

## 设置

- 方法：R²-Gaussian (baseline)、SPAGS (ours)、X-Field (NeurIPS 2025)
- 数据：5 organs × 2/3/4 views，完全同协议
- X-Field 评估：使用 `eval/` 下的最终迭代结果

## 结果

### X-Field PSNR 摘要

| 器官 | 2v | 3v | 4v |
|:----:|:--:|:--:|:--:|
| Chest | 19.57 | 23.94 | 24.45 |
| Head | 22.49 | 25.06 | 26.78 |
| Abdomen | 22.20 | 24.87 | 27.19 |
| Pancreas | 18.12 | 24.94 | 26.36 |
| Foot | 20.45 | 26.03 | 27.25 |
| **平均** | **20.57** | **24.97** | **26.41** |

### SPAGS vs X-Field 对比

| 对比维度 | SPAGS | X-Field |
|:--------:|:-----:|:-------:|
| Avg-3v PSNR | **28.09** | 24.97 (+3.12) |
| Avg-2v PSNR | **21.52** | 20.57 (+0.95) |
| Avg-4v PSNR | **29.19** | 26.41 (+2.78) |
| Gaussian 数量 | ~57K | ~200-500K |
| 渲染速度 | 350+ FPS | 4-21 FPS |

### 完整对比表

详见 `tab_xfield_comparison.tex`。

### 关键发现

- **SPAGS 在所有 15 个 setting 上的 PSNR 都超过 X-Field**，平均超出约 +2.1 dB
- 最大差距出现在 Abdomen 3v（+4.7 dB）和 Chest 3v（+3.3 dB）
- X-Field 的 Gaussian 数约为 SPAGS 的 3.5–8.8 倍，渲染速度慢 15–80 倍
- X-Field 的 SSIM 在部分 setting 上显著落后（如 Abdomen 2v SSIM 0.719 vs SPAGS 0.907）

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想支持线**: Avg-3v SPAGS 排第一 | ✅ SPAGS 28.09 vs X-Field 24.97 |
| **理想支持线**: 3v 至少不落后 | ✅ |
| **理想支持线**: 2v/4v 不被大面积压制 | ✅ SPAGS 在 2v/4v 同样领先 |
| **最低线**: Avg-3v 与 X-Field 打平或小幅落后 | ✅ 大幅领先 |

## 结论

> 与 2025 年后最接近的 direct baseline X-Field 相比，SPAGS 在 same-protocol 2/3/4-view 协议下保持了有竞争力的结果，并在最关键的 3-view 区间体现出最稳定的优势。
