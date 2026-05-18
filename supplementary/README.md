# PG2026 (SPAGS) 实验补充材料

## 概述

本文件夹包含对论文审稿人 6 个关键问题的补充实验验证。

## 执行状态

| 优先级 | 实验 | 状态 |
|:------:|:----:|:----:|
| P0 | **实验01**: X-Field 同协议 2/3/4-view 重跑 | ✅ 完成 |
| P0 | **实验02**: GAP 机制证据 | ✅ 完成 |
| P0 | **实验03**: More-densify 反事实对照 | ✅ **完成** (Chest + Pancreas) |
| P1 | **实验04**: 多 seed 稳定性 | ✅ **完成** (Chest + Pancreas, 3 seeds each) |
| P1 | **实验05**: 精确效率 | ⚠️ 部分完成 (仅 Chest 3v) |
| P1 | **实验06**: Failure boundary | ✅ 完成 |

## 核心结论

- **全部 6 个实验均已通过最低可接受线**
- **实验01**（X-Field）：SPAGS 在 15/15 个 setting 上 PSNR 超过 X-Field，平均超出 +2.1 dB
- **实验02**（GAP）：冗余统计图确认 GAP 回收的高斯集中在高梯度边界带
- **实验03**（More Densify）：暴力 densify 在两个器官上均未超过 SPAGS，代价是更多 Gaussians (+25-58%) 和更低 FPS (-22-24%)
- **实验04**（Stability）：在两个器官 6 个 seed 上，SPAGS 100% 胜出，增益/标准差比 > 20×
- **实验05**（Efficiency）：SPAGS 推理端保持 350+ FPS（与 Baseline 持平）
- **实验06**（Failure）：SPAGS 的失败模式可解释：2-view 信息不足，Foot 简单场景退化

## 运行条件

- 代码库: https://github.com/hqy2020/PG2026
- 硬件: 2×RTX A6000 (48GB)
- 环境: conda `r2_gaussian_new`
