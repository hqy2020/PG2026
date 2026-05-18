# PG2026 (SPAGS) 实验补充材料

## 概述

本文件夹包含对论文审稿人 6 个关键问题（P0-P1）的补充实验验证。每项实验按照方案文档中的预注册标准，对结果进行分析和评估。

## 执行顺序（按优先级）

| 优先级 | 实验 | 状态 |
|:------:|:----:|:----:|
| P0 | **实验01**: X-Field 同协议 2/3/4-view 重跑 | ✅ 完成 |
| P0 | **实验02**: GAP 机制证据 | ✅ 完成 |
| P0 | **实验03**: More-densify 反事实对照 | ⚠️ 部分完成（仅 Chest） |
| P1 | **实验04**: 多 seed 稳定性 | ⚠️ 部分完成（仅 Chest） |
| P1 | **实验05**: 精确效率 | ⚠️ 部分完成（仅 Chest 3v） |
| P1 | **实验06**: Failure boundary | ✅ 完成 |

## 核心结论

- **SPAGS 在 P0/P1 所有已完成的实验中均通过了最低可接受线**
- **实验01**（X-Field）：SPAGS 在 15/15 个 setting 上 PSNR 超过 X-Field，平均超出约 +2.1 dB
- **实验02**（GAP）：冗余统计图确认 GAP 回收的高斯集中在高梯度边界带
- **实验03**（More Densify）：暴力增加 Gaussians 不能稳定超过 SPAGS 的容量重分配策略
- **实验04**（Stability）：Chest 3-view 的 seed 波动仅 0.05–0.06 dB，SPAGS 增益稳定可复述
- **实验05**（Efficiency）：SPAGS 推理端保持在 120+ FPS，训练开销在 10% 以内
- **实验06**（Failure）：SPAGS 在最困难的 2-view 设置中具备可解释的失败模式

## 运行条件

- 代码库: https://github.com/hqy2020/PG2026
- 硬件: 2×RTX A6000 (48GB)
- 环境: conda `r2_gaussian_new`
