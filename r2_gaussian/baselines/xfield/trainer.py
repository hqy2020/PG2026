#
# X-Field training function adapted for r2_gaussian framework
#
# X-Field: A Physically Informed Representation for 3D X-ray Reconstruction
# NeurIPS 2025 Spotlight
#

import os
import os.path as osp
import torch
from random import randint
import numpy as np
import yaml
from tqdm import tqdm

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.unified_logger import get_logger

from .model import XFieldGaussianModel
from .renderer import render_xfield
from .config import XFieldConfig


def training_xfield(
    dataset,
    opt,
    pipe,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    """
    X-Field 训练函数

    Args:
        dataset: ModelParams (数据集参数)
        opt: OptimizationParams (优化参数，合并了 XFieldConfig 参数)
        pipe: PipelineParams (管线参数)
        tb_writer: TensorBoard writer
        testing_iterations: 测试迭代列表
        saving_iterations: 保存迭代列表
        checkpoint_iterations: 检查点迭代列表
        checkpoint: 起始检查点路径
    """
    first_iter = 0

    # 加载场景
    scene = Scene(dataset, shuffle=False)

    # 获取扫描仪配置
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])

    # 获取 X-Field 配置
    xf_config = XFieldConfig()

    # 尺度参数
    max_scale = (
        getattr(opt, 'max_scale', None) or xf_config.densify_scale_threshold
    )
    max_scale = max_scale * volume_to_world if max_scale else None

    densify_scale_threshold = (
        getattr(opt, 'densify_scale_threshold', None) or xf_config.densify_scale_threshold
    )
    if densify_scale_threshold:
        densify_scale_threshold = densify_scale_threshold * volume_to_world

    # 尺度边界（用于 Softplus density 约束）
    scale_bound = None
    scale_min = getattr(opt, 'scale_min', xf_config.scale_min)
    scale_max = getattr(opt, 'scale_max', xf_config.scale_max)
    if scale_min > 0 and scale_max > 0:
        scale_bound = np.array([scale_min, scale_max]) * volume_to_world

    # 创建 X-Field 模型
    gaussians = XFieldGaussianModel(scale_bound)

    # 加载初始化点云
    if dataset.ply_path and osp.exists(dataset.ply_path):
        gaussians.create_from_r2_init(dataset.ply_path, spatial_lr_scale=1.0)
    else:
        raise ValueError(
            f"X-Field requires initialization file. "
            f"Please run initialize_pcd.py first or provide --ply_path"
        )

    scene.gaussians = gaussians

    # 设置优化参数（合并 XFieldConfig 和传入参数）
    class XFieldOptParams:
        def __init__(self, opt, xf_config):
            self.position_lr_init = getattr(
                opt, 'position_lr_init', xf_config.position_lr_init
            )
            self.position_lr_final = getattr(
                opt, 'position_lr_final', xf_config.position_lr_final
            )
            self.position_lr_max_steps = getattr(
                opt, 'position_lr_max_steps', xf_config.position_lr_max_steps
            )
            self.density_lr_init = getattr(
                opt, 'density_lr_init', xf_config.density_lr_init
            )
            self.density_lr_final = getattr(
                opt, 'density_lr_final', xf_config.density_lr_final
            )
            self.density_lr_max_steps = getattr(
                opt, 'density_lr_max_steps', xf_config.density_lr_max_steps
            )
            self.scaling_lr_init = getattr(
                opt, 'scaling_lr_init', xf_config.scaling_lr_init
            )
            self.scaling_lr_final = getattr(
                opt, 'scaling_lr_final', xf_config.scaling_lr_final
            )
            self.scaling_lr_max_steps = getattr(
                opt, 'scaling_lr_max_steps', xf_config.scaling_lr_max_steps
            )
            self.rotation_lr_init = getattr(
                opt, 'rotation_lr_init', xf_config.rotation_lr_init
            )
            self.rotation_lr_final = getattr(
                opt, 'rotation_lr_final', xf_config.rotation_lr_final
            )
            self.rotation_lr_max_steps = getattr(
                opt, 'rotation_lr_max_steps', xf_config.rotation_lr_max_steps
            )

    xf_opt = XFieldOptParams(opt, xf_config)
    gaussians.training_setup(xf_opt)

    logger = get_logger()

    # 加载检查点
    if checkpoint is not None:
        state, first_iter = torch.load(checkpoint)
        gaussians.restore(state, xf_opt)
        logger.config(f"Loaded X-Field checkpoint from {checkpoint}")

    # 密集化参数
    densify_from_iter = getattr(
        opt, 'densify_from_iter', xf_config.densify_from_iter
    )
    densify_until_iter = getattr(
        opt, 'densify_until_iter', xf_config.densify_until_iter
    )
    densification_interval = getattr(
        opt, 'densification_interval', xf_config.densification_interval
    )
    densify_grad_threshold = getattr(
        opt, 'densify_grad_threshold', xf_config.densify_grad_threshold
    )
    min_density = getattr(
        opt, 'min_density', xf_config.min_density
    )
    lambda_dssim = getattr(
        opt, 'lambda_dssim', xf_config.lambda_dssim
    )
    max_num_gaussians = getattr(
        opt, 'max_num_gaussians', xf_config.max_num_gaussians
    )

    # ==================== 训练循环 ====================
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="X-Field Training"
    )
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 随机选择视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )

        # 渲染
        render_pkg = render_xfield(viewpoint_cam, gaussians, pipe)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # GT 图像
        gt_image = viewpoint_cam.original_image.to("cuda")

        # 计算损失
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Pts": f"{gaussians.get_num_points()}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard 日志
            if tb_writer:
                tb_writer.add_scalar(
                    "train/loss_l1", Ll1.item(), iteration
                )
                tb_writer.add_scalar(
                    "train/loss_total", loss.item(), iteration
                )
                tb_writer.add_scalar(
                    "train/total_points",
                    gaussians.get_num_points(),
                    iteration,
                )

            # 密集化控制
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )
            gaussians.add_densification_stats(
                viewspace_point_tensor, visibility_filter
            )

            if iteration < densify_until_iter:
                if (
                    iteration > densify_from_iter
                    and iteration % densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        densify_grad_threshold,
                        min_density,
                        None,  # max_screen_size
                        max_scale,
                        max_num_gaussians,
                        densify_scale_threshold,
                        iteration,
                        bbox,
                    )

            # 检查高斯是否全部消失
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # 优化步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 保存模型
            if iteration in saving_iterations or iteration == opt.iterations:
                logger.info(
                    "Saving X-Field model", iteration=iteration
                )
                save_path = osp.join(
                    scene.model_path, f"xfield_iter_{iteration}.pth"
                )
                torch.save(
                    (gaussians.capture(), iteration), save_path
                )
                # 同时保存 PLY
                gaussians.save_ply(
                    osp.join(
                        scene.model_path,
                        f"point_cloud/iteration_{iteration}/point_cloud.ply",
                    )
                )

            # 评估
            if iteration in testing_iterations:
                _xfield_eval(
                    tb_writer, iteration, scene, gaussians, pipe
                )

            # 检查点
            if iteration in checkpoint_iterations:
                logger.info(
                    "Saving Checkpoint", iteration=iteration
                )
                ckpt_path = osp.join(
                    scene.model_path, f"chkpnt_xfield_{iteration}.pth"
                )
                torch.save(
                    (gaussians.capture(), iteration), ckpt_path
                )


def _xfield_eval(tb_writer, iteration, scene, gaussians, pipe):
    """X-Field 评估函数"""
    eval_save_path = osp.join(
        scene.model_path, "eval", f"iter_{iteration:06d}"
    )
    os.makedirs(eval_save_path, exist_ok=True)
    torch.cuda.empty_cache()

    # 2D 评估（新视角合成投影）
    validation_configs = [
        {"name": "render_train", "cameras": scene.getTrainCameras()},
        {"name": "render_test", "cameras": scene.getTestCameras()},
    ]

    psnr_2d, ssim_2d = None, None
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            images = []
            gt_images = []

            for idx, viewpoint in enumerate(config["cameras"]):
                render_result = render_xfield(viewpoint, gaussians, pipe)
                image = render_result["render"]
                gt_image = viewpoint.original_image.to("cuda")

                images.append(image)
                gt_images.append(gt_image)

            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)

            psnr_2d, psnr_2d_projs = metric_proj(
                gt_images, images, "psnr"
            )
            ssim_2d, ssim_2d_projs = metric_proj(
                gt_images, images, "ssim"
            )

            eval_dict_2d = {
                "psnr_2d": float(psnr_2d),
                "ssim_2d": float(ssim_2d),
                "psnr_2d_projs": [float(p) for p in psnr_2d_projs],
                "ssim_2d_projs": [float(s) for s in ssim_2d_projs],
            }
            with open(
                osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                "w",
            ) as f:
                yaml.dump(
                    eval_dict_2d,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

            if tb_writer:
                tb_writer.add_scalar(
                    f"xfield/{config['name']}/psnr_2d",
                    psnr_2d,
                    iteration,
                )
                tb_writer.add_scalar(
                    f"xfield/{config['name']}/ssim_2d",
                    ssim_2d,
                    iteration,
                )

    # 3D 体积评估（体素重建）
    try:
        from xray_gaussian_rasterization_voxelization import (
            GaussianVoxelizationSettings,
            GaussianVoxelizer,
        )

        scanner_cfg = scene.scanner_cfg
        offOrigin = scanner_cfg["offOrigin"]
        nVoxel = scanner_cfg["nVoxel"]
        sVoxel = scanner_cfg["sVoxel"]

        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=1.0,
            nVoxel_x=int(nVoxel[0]),
            nVoxel_y=int(nVoxel[1]),
            nVoxel_z=int(nVoxel[2]),
            sVoxel_x=float(sVoxel[0]),
            sVoxel_y=float(sVoxel[1]),
            sVoxel_z=float(sVoxel[2]),
            center_x=float(offOrigin[0]),
            center_y=float(offOrigin[1]),
            center_z=float(offOrigin[2]),
            prefiltered=False,
            debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

        vol_pred, _ = voxelizer(
            means3D=gaussians.get_xyz,
            opacities=gaussians.get_density,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            cov3D_precomp=None,
        )

        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")

        eval_dict_3d = {
            "psnr_3d": float(psnr_3d),
            "ssim_3d": float(ssim_3d),
        }
        with open(
            osp.join(eval_save_path, "eval3d_xfield.yml"), "w"
        ) as f:
            yaml.dump(
                eval_dict_3d, f, default_flow_style=False, sort_keys=False
            )

        if tb_writer:
            tb_writer.add_scalar(
                "xfield/psnr_3d", psnr_3d, iteration
            )
            tb_writer.add_scalar(
                "xfield/ssim_3d", ssim_3d, iteration
            )

        logger = get_logger()
        logger.eval(
            f"X-Field [{iteration}]: "
            f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, "
            f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}",
            iteration=iteration,
        )
    except ImportError:
        logger = get_logger()
        logger.warning(
            "Skipping 3D evaluation: "
            "xray_gaussian_rasterization_voxelization not available"
        )

    torch.cuda.empty_cache()
