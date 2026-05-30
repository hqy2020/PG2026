#
# X-Field renderer for r2_gaussian framework
#
# Uses xfield_rasterization CUDA kernel for X-ray projection rendering.
# X-Field uses density (Softplus activated) instead of opacity (sigmoid).
# Single-channel grayscale output (no spherical harmonics).
#

import torch
import math
from typing import Dict

from xfield_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .model import XFieldGaussianModel
from ..registry import BaseRenderer


def render_xfield(
    viewpoint_camera,
    pc: XFieldGaussianModel,
    pipe,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    X-Field 渲染函数

    使用 xfield_rasterization 进行 X-ray 投影渲染。
    X-Field 的 density 直接作为光栅化器的 opacity 输入。

    Args:
        viewpoint_camera: 相机视角
        pc: XFieldGaussianModel 实例
        pipe: PipelineParams
        bg_color: 背景颜色 (可选，X-Field 始终使用黑色背景)
        scaling_modifier: 尺度修正因子

    Returns:
        dict:
            - render: 渲染的 X-ray 投影 [1, H, W]
            - viewspace_points: 屏幕空间点 (用于梯度)
            - visibility_filter: 可见性掩码
            - radii: 2D 半径
    """
    # 创建屏幕空间点张量（用于获取梯度）
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置相机参数
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError(f"Unsupported camera mode: {mode}")

    # 光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯参数
    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density  # Softplus 激活后的密度值

    # 处理协方差
    scales = None
    rotations = None
    cov3D_precomp = None
    if hasattr(pipe, 'compute_cov3D_python') and pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 渲染（X-Field 将 density 作为 opacities 传入）
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


class XFieldRenderer(BaseRenderer):
    """X-Field 渲染器类"""

    def __init__(self, pipe=None):
        self.pipe = pipe

    def render(
        self,
        viewpoint,
        model: XFieldGaussianModel,
        pipe=None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """渲染接口实现"""
        if pipe is None:
            pipe = self.pipe
        return render_xfield(viewpoint, model, pipe, **kwargs)
