#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import sys
import torch
from tqdm import tqdm, trange
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser
from random import randint
import SimpleITK as sitk

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.image_utils import metric_vol, metric_proj

# ─────────── 方法加载字典 ───────────

def _load_model_r2_gaussian(dataset, pipeline, iteration, full_args):
    """加载 R²-Gaussian / XRA-GS 模型"""
    gaussians = GaussianModel(None, args=full_args if full_args is not None else dataset)
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    num_train_views = len(dataset.train_cameras) if hasattr(dataset, 'train_cameras') else 0
    if num_train_views > 0:
        gaussians.set_num_train_views(num_train_views)
    gaussians.current_iteration = loaded_iter
    return gaussians, render, query, loaded_iter


def _load_model_xgaussian(dataset, pipeline, iteration, scene):
    """加载 X-Gaussian 模型"""
    from r2_gaussian.baselines.xgaussian import XGaussianModel, render_xgaussian
    from r2_gaussian.baselines.xgaussian.renderer import query_xgaussian

    ckpt_path = osp.join(dataset.model_path, f"xgaussian_iter_{iteration}.pth")
    alt_path = osp.join(dataset.model_path, f"chkpnt_xgaussian_{iteration}.pth")
    if not osp.exists(ckpt_path):
        if osp.exists(alt_path):
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(
                f"Cannot find X-Gaussian checkpoint for iteration {iteration}"
            )

    state, loaded_iter = torch.load(ckpt_path)
    gaussians = XGaussianModel()
    gaussians.active_sh_degree = state.get('active_sh_degree', 0)
    gaussians._xyz = state['_xyz']
    gaussians._features_dc = state['_features_dc']
    gaussians._features_rest = state['_features_rest']
    gaussians._scaling = state['_scaling']
    gaussians._rotation = state['_rotation']
    gaussians._opacity = state['_opacity']
    gaussians.max_radii2D = state['max_radii2D']
    gaussians.spatial_lr_scale = state.get('spatial_lr_scale', 1.0)
    return gaussians, render_xgaussian, query_xgaussian, loaded_iter


def _load_model_corgs(dataset, pipeline, iteration, scene):
    """加载 CoR-GS 模型（使用主场 gs0）"""
    from r2_gaussian.baselines.xgaussian import XGaussianModel, render_xgaussian
    from r2_gaussian.baselines.xgaussian.renderer import query_xgaussian

    ckpt_path = osp.join(dataset.model_path, f"corgs_iter_{iteration}.pth")
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find CoR-GS checkpoint for iteration {iteration}")

    checkpoint = torch.load(ckpt_path)
    state = checkpoint["gs0"]
    loaded_iter = checkpoint.get("iteration", iteration)

    gaussians = XGaussianModel()
    gaussians.active_sh_degree = state.get('active_sh_degree', 0)
    gaussians._xyz = state['_xyz']
    gaussians._features_dc = state['_features_dc']
    gaussians._features_rest = state['_features_rest']
    gaussians._scaling = state['_scaling']
    gaussians._rotation = state['_rotation']
    gaussians._opacity = state['_opacity']
    gaussians.max_radii2D = state['max_radii2D']
    gaussians.spatial_lr_scale = state.get('spatial_lr_scale', 1.0)
    return gaussians, render_xgaussian, query_xgaussian, loaded_iter


def _load_model_fsgs(dataset, pipeline, iteration, scene):
    """加载 FSGS 模型"""
    from r2_gaussian.baselines.fsgs import FSGSModel, render_fsgs, query_fsgs

    ckpt_path = osp.join(dataset.model_path, f"fsgs_iter_{iteration}.pth")
    alt_path = osp.join(dataset.model_path, f"chkpnt_fsgs_{iteration}.pth")
    if not osp.exists(ckpt_path):
        if osp.exists(alt_path):
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find FSGS checkpoint for iteration {iteration}")

    state, loaded_iter = torch.load(ckpt_path)
    gaussians = FSGSModel()
    gaussians.active_sh_degree = state.get('active_sh_degree', 0)
    gaussians._xyz = state['_xyz']
    gaussians._features_dc = state['_features_dc']
    gaussians._features_rest = state['_features_rest']
    gaussians._scaling = state['_scaling']
    gaussians._rotation = state['_rotation']
    gaussians._opacity = state['_opacity']
    gaussians.max_radii2D = state['max_radii2D']
    gaussians.spatial_lr_scale = state.get('spatial_lr_scale', 1.0)
    gaussians.confidence = state.get('confidence', None)
    gaussians.init_point = state.get('init_point', None)
    return gaussians, render_fsgs, query_fsgs, loaded_iter


def _load_model_dngaussian(dataset, pipeline, iteration, scene):
    """加载 DNGaussian 模型"""
    from r2_gaussian.baselines.dngaussian import DNGaussianModel, render_dngaussian, query_dngaussian

    ckpt_path = osp.join(dataset.model_path, f"dngaussian_iter_{iteration}.pth")
    alt_path = osp.join(dataset.model_path, f"chkpnt_dngaussian_{iteration}.pth")
    if not osp.exists(ckpt_path):
        if osp.exists(alt_path):
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find DNGaussian checkpoint for iteration {iteration}")

    state, loaded_iter = torch.load(ckpt_path)
    gaussians = DNGaussianModel()
    gaussians._xyz = state['_xyz']
    gaussians._scaling = state['_scaling']
    gaussians._rotation = state['_rotation']
    gaussians._opacity = state['_opacity']
    gaussians.max_radii2D = state['max_radii2D']
    gaussians.spatial_lr_scale = state.get('spatial_lr_scale', 1.0)
    if 'neural_renderer_state' in state:
        gaussians._init_neural_renderer()
        gaussians.neural_renderer.load_state_dict(state['neural_renderer_state'])
    return gaussians, render_dngaussian, query_dngaussian, loaded_iter


def _load_model_xfield(dataset, pipeline, iteration, scene):
    """加载 X-Field 模型"""
    from r2_gaussian.baselines.xfield import XFieldGaussianModel, render_xfield

    ckpt_path = osp.join(dataset.model_path, f"xfield_iter_{iteration}.pth")
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find X-Field checkpoint for iteration {iteration}")

    state, loaded_iter = torch.load(ckpt_path)
    gaussians = XFieldGaussianModel()
    gaussians._xyz = state['xyz']
    gaussians._scaling = state['scaling']
    gaussians._rotation = state['rotation']
    gaussians._density = state['density']
    gaussians.max_radii2D = state.get('max_radii2D', torch.zeros(gaussians._xyz.shape[0], device="cuda"))
    gaussians.spatial_lr_scale = state.get('spatial_lr_scale', 1.0)

    # X-Field 没有 query 函数，使用 voxelization 进行 3D 评估
    def _query_xfield(gaussians, offOrigin, nVoxel, sVoxel, pipe):
        from xray_gaussian_rasterization_voxelization import (
            GaussianVoxelizationSettings, GaussianVoxelizer,
        )
        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=1.0,
            nVoxel_x=int(nVoxel[0]), nVoxel_y=int(nVoxel[1]), nVoxel_z=int(nVoxel[2]),
            sVoxel_x=float(sVoxel[0]), sVoxel_y=float(sVoxel[1]), sVoxel_z=float(sVoxel[2]),
            center_x=float(offOrigin[0]), center_y=float(offOrigin[1]), center_z=float(offOrigin[2]),
            prefiltered=False, debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)
        vol_pred, radii = voxelizer(
            means3D=gaussians.get_xyz,
            opacities=gaussians.get_density,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            cov3D_precomp=None,
        )
        return {"vol": vol_pred, "radii": radii}

    return gaussians, render_xfield, _query_xfield, loaded_iter


# ─────────── 方法加载注册表 ───────────
METHOD_LOADERS = {
    "r2_gaussian": _load_model_r2_gaussian,
    "xgaussian": _load_model_xgaussian,
    "corgs": _load_model_corgs,
    "fsgs": _load_model_fsgs,
    "dngaussian": _load_model_dngaussian,
    "xfield": _load_model_xfield,
}


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
    full_args=None,
    method="r2_gaussian",
):
    """统一测试入口，支持多种方法"""
    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up model (方法分发)
    if method == "r2_gaussian":
        # R²-Gaussian / XRA-GS: 使用原始加载流程（PLY + ADM）
        gaussians = GaussianModel(None, args=full_args if full_args is not None else dataset)
        loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
        scene.gaussians = gaussians
        num_train_views = len(scene.getTrainCameras())
        if num_train_views > 0:
            gaussians.set_num_train_views(num_train_views)
        gaussians.current_iteration = loaded_iter
        render_func = render
        query_func = query
    else:
        # 其他 baseline: 从 checkpoint 加载
        loader = METHOD_LOADERS.get(method)
        if loader is None:
            raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_LOADERS.keys())}")
        gaussians, render_func, query_func, loaded_iter = loader(dataset, pipeline, iteration, scene)
        scene.gaussians = gaussians

    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    if not skip_render_train:
        evaluate_render(
            save_path, "render_train",
            scene.getTrainCameras(), gaussians, pipeline,
            render_func=render_func,
        )
    # Evaluate projection test
    if not skip_render_test:
        evaluate_render(
            save_path, "render_test",
            scene.getTestCameras(), gaussians, pipeline,
            render_func=render_func,
        )
    # Evaluate volume reconstruction
    if not skip_recon:
        evaluate_volume(
            save_path, "reconstruction",
            scene.scanner_cfg, gaussians, pipeline,
            scene.vol_gt, query_func=query_func,
        )


def evaluate_volume(
    save_path, name, scanner_cfg, gaussians, pipeline, vol_gt, query_func=None,
):
    """Evaluate volume reconstruction."""
    if query_func is None:
        from r2_gaussian.gaussian import query as query_func

    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    query_pkg = query_func(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline,
    )
    vol_pred = query_pkg["vol"]

    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")

    multithread_write(
        [vol_gt[..., i][None] for i in range(vol_gt.shape[2])],
        slice_save_path, "_gt",
    )
    multithread_write(
        [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
        slice_save_path, "_pred",
    )
    eval_dict = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
    }

    with open(osp.join(save_path, "eval3d.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    np.save(osp.join(save_path, "vol_gt.npy"), t2a(vol_gt))
    np.save(osp.join(save_path, "vol_pred.npy"), t2a(vol_pred))
    sitk.WriteImage(
        sitk.GetImageFromArray(t2a(vol_gt).transpose(2, 0, 1)),
        os.path.join(save_path, "vol_gt.nii.gz"),
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(t2a(vol_pred).transpose(2, 0, 1)),
        os.path.join(save_path, "vol_pred.nii.gz"),
    )

    print(f"{name} complete. psnr_3d: {psnr_3d}, ssim_3d: {ssim_3d}")


def evaluate_render(save_path, name, views, gaussians, pipeline, render_func=None):
    """Evaluate projection rendering with timing."""
    if render_func is None:
        from r2_gaussian.gaussian import render as render_func

    proj_save_path = osp.join(save_path, name)

    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    render_times = []

    for view in tqdm(views, desc="render {}".format(name), leave=False):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        rendering = render_func(view, gaussians, pipeline)["render"]
        end.record()
        torch.cuda.synchronize()
        render_times.append(start.elapsed_time(end))
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)

    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
    }
    with open(osp.join(save_path, "eval2d_{}.yml".format(name)), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    total_ms = sum(render_times)
    avg_ms = total_ms / len(render_times) if render_times else 0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    timing_dict = {
        "num_views": len(views),
        "total_render_time_ms": round(total_ms, 2),
        "avg_render_time_per_view_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
    }
    with open(osp.join(save_path, "timing_{}.yml".format(name)), "w") as f:
        yaml.dump(timing_dict, f, default_flow_style=False, sort_keys=False)

    print(
        "{} complete. psnr_2d: {}, ssim_2d: {}, "
        "avg_render: {:.1f}ms, fps: {:.1f}.".format(
            name, eval_dict["psnr_2d"], eval_dict["ssim_2d"], avg_ms, fps
        )
    )


def multithread_write(image_list, path, suffix):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
            )
            np.save(
                osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
                image.cpu().numpy()[0],
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--method", type=str, default="r2_gaussian",
                        choices=["r2_gaussian", "xgaussian", "corgs", "fsgs",
                                 "dngaussian", "xfield"],
                        help="选择评估方法")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
            full_args=args,
            method=args.method,
        )
