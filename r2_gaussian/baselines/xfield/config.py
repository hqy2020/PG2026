#
# X-Field configuration parameters
#
# Adapted from X-Field (NeurIPS 2025 Spotlight)
#

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class XFieldConfig:
    """X-Field 特有参数配置

    参考 X-Field: https://github.com/Brack-Wang/X-Field
    """

    # === 尺度边界（用于 density Softplus 约束） ===
    scale_min: float = 0.0005
    scale_max: float = 0.5

    # === 密集化参数 ===
    densify_from_iter: int = 500
    densify_until_iter: int = 8000
    densification_interval: int = 100
    densify_grad_threshold: float = 3.0e-5   # X-Field 使用更小的梯度阈值
    densify_scale_threshold: float = 0.1
    densify_gap_start: int = 1500            # GAP densification 起始迭代

    # === Density 控制 ===
    min_density: float = 0.00001

    # === 学习率 ===
    position_lr_init: float = 0.0002
    position_lr_final: float = 0.00002
    position_lr_max_steps: int = 5000
    density_lr_init: float = 0.008
    density_lr_final: float = 0.001
    density_lr_max_steps: int = 5000
    scaling_lr_init: float = 0.005
    scaling_lr_final: float = 0.0005
    scaling_lr_max_steps: int = 5000
    rotation_lr_init: float = 0.001
    rotation_lr_final: float = 0.0001
    rotation_lr_max_steps: int = 5000

    # === 损失权重 ===
    lambda_dssim: float = 0.1

    # === 高斯数量上限 ===
    max_num_gaussians: int = 500_000


def get_xfield_config() -> XFieldConfig:
    """获取默认 X-Field 配置"""
    return XFieldConfig()
