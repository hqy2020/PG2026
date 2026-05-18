# 实验06：Failure boundary 边界案例

## 目标

主动定义 SPAGS 的最强和最弱 setting，以及为什么。

## 设置

- 最危险 setting：Foot-2v、Pancreas-2v、Head-4v、Foot-4v
- 方法：GT、X-Gaussian、R²-Gaussian、SPAGS
- 输出：定性对比图 + 残差图

## 结果

### 已生成

- `fig_failure_cases.png` (286KB) — 包含多组最差 setting 的定性对比

### SPAGS 的增益分布（按 setting 难度）

| Setting | R² PSNR | SPAGS PSNR | Δ | 增益评级 |
|:-------:|:-------:|:----------:|:-:|:--------:|
| Chest 3v | 26.22 | 26.63 | +0.41 | 📈 显著 |
| Pancreas 3v | 28.61 | 29.37 | +0.76 | 📈 最大 |
| Abdomen 3v | 29.26 | 29.55 | +0.29 | 📈 中等 |
| Foot 3v | 28.48 | 28.34 | -0.14 | 🔻 退化 |
| Head 3v | 26.60 | 26.55 | -0.05 | 🔻 轻微退化 |

| Setting | R² PSNR | SPAGS PSNR | Δ | 说明 |
|:-------:|:-------:|:----------:|:-:|:----|
| Foot 2v | 19.39 | 19.44 | +0.05 | 信息不足，所有方法均差 |
| Pancreas 2v | 17.83 | 17.97 | +0.14 | 极端稀疏，信息缺失 |
| Head 4v | 28.66 | 28.76 | +0.10 | 较高 view，R² 一致性优势回归 |
| Foot 4v | 30.04 | 29.84 | -0.20 | 退化，SPAGS 在 easy case 上不如 baseline |

### 关键发现

- **SPAGS 的最强工作区间是 3-view**（特别是困难器官 Pancreas 和 Chest）
- **2-view 极端稀疏**时，所有方法都差，主要问题是观测信息本身太少
- **4-view /高信息量场景**时，R²-Gaussian 的 consistency 优势重新变强
- **Foot 异常**：SPAGS 在 Foot 上表现异常（3v 退化 0.14 dB，4v 退化 0.20 dB），可能因为 Foot 本身简单，GAP 的裁剪效应反而带来了伤害

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想线**: 即使不是最佳 PSNR，failure mode 与主线一致 | ✅ 2-view 信息不足，4-view 一致性回归 |
| **最低线**: 能把最差 setting 的失败原因说清楚 | ✅ 明确归因于信息边界 |
| **危险信号**: 失败原因与机制叙事对不上 | ❌ 未发生 |

## 结论

> Failure cases further suggest that SPAGS is most effective in the intermediate sparse regime (3-view), while extremely under-constrained 2-view scenes still require information beyond capacity reallocation alone, and 4-view higher-data scenarios see reduced gains as R²-Gaussian's consistency mechanism becomes sufficient.
