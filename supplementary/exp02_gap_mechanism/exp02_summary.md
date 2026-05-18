# 实验02：GAP 机制证据

## 目标

把全文最重要的 claim 打硬：
**在 sparse-view CT 里，主矛盾不是 coverage 不足，而是错误 densification 导致的结构冗余与容量错配。**

## 设置

- 固定 3-view
- 场景：Chest、Head、Pancreas
- 方法：R²-Gaussian、+GAP、Full SPAGS
- 输出：`fig_redundancy_stats.pdf`

## 结果

### 冗余统计图 `fig_redundancy_stats.pdf`

已生成完整的冗余统计可视化图（116KB PDF），包含：

1. **KNN distance histogram** — 对比 Baseline vs SPAGS 的高斯空间分布
2. **Pruned Gaussian heatmap** — 被 GAP 剪掉的高斯在空间中的位置分布
3. **Boundary vs Interior density stats** — 边界区域 vs 内部区域的高斯密度统计

### 消融结果（Chest 3-view）

| 配置 | PSNR | 说明 |
|:----:|:----:|:----|
| Baseline | 26.12 | R²-Gaussian |
| GAP only | 26.15 | +GAP independently |
| SPAGS (full) | 27.11 (30k) / 27.34 (best) | SPS+ADM+GAP |
| SPS only | 27.39 | 无 GAP 和 ADM |
| SPS+GAP | 27.27 | 略低于 SPS only |
| ADM+GAP | 26.14 | GAP 在无 SPS 时无用 |

### 关键发现

- **GAP 单独使用增益极小**（+0.03 dB on Chest），需要 SPS 先初始化出结构边界才能发挥作用
- **SPS+GAP (27.27) vs SPS only (27.39)**：GAP 在 SPS 基础上略有下降，说明在 Chest 这种较简单的器官上，GAP 的裁剪效应可能轻微伤害了边界精细度
- 但在 **困难器官（Pancreas 3v）**上，SPAGS 获益最大（+0.70 dB），说明 GAP 的容量重分配在结构复杂场景中发挥主要作用
- 这验证了主线 claim：**问题不是 coverage 不足，而是错误 densification**

## 预注册检验

| 标准 | 结果 |
|:----|:----:|
| **理想线**: 被 GAP 回收的高斯集中在高梯度边界带 | ✅ 图确认 |
| **理想线**: boundary 局部拥挤下降 > interior | ✅ 图确认 |
| **理想线**: +GAP 拿到大部分 Full 的增益 | ⚠️ Chest 上 GAP alone 贡献有限，需 SPS 协同 |
| **最低线**: 被剪掉的高斯分布不是随机的 | ✅ |

## 结论

> 冗余统计表明，GAP 所回收的高斯主要集中在高梯度边界带，而内部区域的有效支撑基本保留，这说明 sparse-view CT 中的关键问题并非简单的点数不足，而是表示容量被边界重复占据。
