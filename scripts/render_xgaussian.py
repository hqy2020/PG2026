#!/usr/bin/env python3
"""
Render X-Gaussian - minimal approach.
No argparse, just direct object creation.
"""
import os, sys, glob, yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/qyhu/Documents/r2_ours/PG2026")


def render_xgaussian_organ(organ):
    BASE = "/home/qyhu/Documents/r2_ours/PG2026"
    
    dirs = glob.glob(f"{BASE}/output/*{organ}*3v*xgaussian*")
    dirs = [d for d in dirs if 'opt_' not in os.path.basename(d)]
    if not dirs:
        print(f"  ❌ {organ}: no dir")
        return False
    
    model_path = dirs[0]
    source_path = f"{BASE}/data/369/{organ}_50_3views.pickle"
    output_iter = 30000
    test_dir = f"{model_path}/test/iter_{output_iter}/render_test"
    
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 10:
        print(f"  ✅ {organ}: already done")
        return True
    
    print(f"  🔄 {organ}...")
    
    from r2_gaussian.arguments import ModelParams, PipelineParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.baselines.xgaussian.model import XGaussianModel
    from r2_gaussian.baselines.xgaussian.renderer import render_xgaussian
    from r2_gaussian.utils.general_utils import safe_state
    from r2_gaussian.utils.image_utils import metric_proj
    
    safe_state(True)
    device = torch.device("cuda")
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    dataset = ModelParams(parser)
    dataset.source_path = source_path
    dataset.model_path = model_path
    dataset.data_device = "cuda"
    dataset.eval = True
    
    # Load init file path from cfg_args
    if os.path.exists(f"{model_path}/cfg_args"):
        with open(f"{model_path}/cfg_args") as f:
            content = f.read()
        import ast
        # Simple parse: extract ply_path value
        if "ply_path=" in content:
            start = content.index("ply_path=") + 9
            end = content.index(",", start) if "," in content[start:] else content.index(")", start)
            val = content[start:end].strip().strip("'\"")
            if val and val != "''" and val != '""':
                dataset.ply_path = val
        # Extract scale_min/scale_max
        for key in ['scale_min', 'scale_max']:
            if f"{key}=" in content:
                start = content.index(f"{key}=") + len(key) + 1
                end = content.index(",", start) if "," in content[start:] else content.index(")", start)
                val = content[start:end].strip()
                try:
                    setattr(dataset, key, float(val))
                except:
                    pass
    
    # Create PipelineParams
    pipe = PipelineParams(parser)
    
    # Create scene
    scene = Scene(dataset, shuffle=False)
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print(f"  ❌ {organ}: no test cameras")
        return False
    print(f"     {len(test_cameras)} test cameras")
    
    # Create model
    gaussians = XGaussianModel(sh_degree=3, scale_bound=None)
    
    # Initialize from R2 init
    if dataset.ply_path and os.path.exists(dataset.ply_path):
        gaussians.create_from_r2_init(dataset.ply_path, spatial_lr_scale=1.0)
    else:
        init_path = dataset.ply_path or f"{BASE}/data/369/init_{organ}_50_3views.npy"
        if os.path.exists(init_path):
            gaussians.create_from_r2_init(init_path, spatial_lr_scale=1.0)
            print(f"     init from {init_path}")
        else:
            print(f"  ❌ {organ}: no init file")
            return False
    
    print(f"     init: {gaussians.get_xyz.shape[0]} points")
    
    # Load checkpoint
    ckpt = torch.load(f"{model_path}/xgaussian_iter_{output_iter}.pth", map_location="cpu")
    state_dict = ckpt[0] if isinstance(ckpt, tuple) else ckpt
    
    # Restore
    for key in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
        if key in state_dict:
            getattr(gaussians, key).data = state_dict[key].to(device)
    gaussians.active_sh_degree = state_dict.get('active_sh_degree', 3)
    if hasattr(gaussians, 'max_radii2D'):
        gaussians.max_radii2D = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    if hasattr(gaussians, 'denom'):
        gaussians.denom = torch.ones(gaussians.get_xyz.shape[0], device=device)
    if 'spatial_lr_scale' in state_dict:
        gaussians.spatial_lr_scale = state_dict['spatial_lr_scale']
    
    # Move to GPU
    for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
                 'max_radii2D', 'denom', 'xyz_gradient_accum']:
        if hasattr(gaussians, attr) and isinstance(getattr(gaussians, attr), torch.Tensor):
            setattr(gaussians, attr, getattr(gaussians, attr).to(device))
    
    # Render
    os.makedirs(test_dir, exist_ok=True)
    render_list, gt_list = [], []
    
    for idx, camera in enumerate(tqdm(test_cameras, desc=f"X-G {organ}", leave=False)):
        camera = camera.to(device)
        try:
            render_pkg = render_xgaussian(camera, gaussians, pipe)
            rendering = render_pkg["render"]
            gt = camera.original_image.to(device)
            
            pred_img = rendering[0].detach().cpu().numpy()
            gt_img = gt[0].detach().cpu().numpy()
            pred_img = np.clip(pred_img, 0, 1)
            gt_img = np.clip(gt_img, 0, 1)
            
            Image.fromarray((pred_img * 255).astype(np.uint8)).save(f"{test_dir}/{idx:05d}_pred.png")
            Image.fromarray((gt_img * 255).astype(np.uint8)).save(f"{test_dir}/{idx:05d}_gt.png")
            render_list.append(rendering[0])
            gt_list.append(gt[0])
        except Exception as e:
            print(f"     view {idx}: {e}")
    
    if render_list:
        images = torch.stack(render_list, dim=0)
        gt_images = torch.stack(gt_list, dim=0)
        psnr_2d, psnr_projs = metric_proj(gt_images, images, "psnr")
        ssim_2d, ssim_projs = metric_proj(gt_images, images, "ssim")
        
        eval_dir = f"{model_path}/eval/iter_{output_iter:06d}"
        os.makedirs(eval_dir, exist_ok=True)
        with open(f"{eval_dir}/eval2d_xgaussian.yml", "w") as f:
            yaml.dump({"psnr_2d": float(psnr_2d), "ssim_2d": float(ssim_2d),
                      "psnr_2d_projs": [float(p) for p in psnr_projs],
                      "ssim_2d_projs": [float(p) for p in ssim_projs]},
                     f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✅ {organ}: PSNR={psnr_2d:.2f}")
    else:
        print(f"  ❌ {organ}: no renders")
    
    return True


if __name__ == "__main__":
    for organ in ["chest", "head", "abdomen", "foot", "pancreas"]:
        render_xgaussian_organ(organ)
    print("\n✅ Done!")
