#
# X-Field GaussianModel adapted for r2_gaussian framework
#
# X-Field: A Physically Informed Representation for 3D X-ray Reconstruction
# NeurIPS 2025 Spotlight
#
# Core differences from vanilla 3DGS:
#   - Uses density (Softplus activation) instead of opacity (sigmoid)
#   - No spherical harmonics (grayscale single-channel output)
#   - Has densify_gap: density gradient-aware densification
#   - Supports scale_bound for density scale constraints
#

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from plyfile import PlyData, PlyElement

from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    inverse_softplus,
    get_expon_lr_func,
    build_rotation,
    strip_symmetric,
    build_scaling_rotation,
)
from simple_knn._C import distCUDA2

from ..registry import GaussianBaseModel


class XFieldGaussianModel(GaussianBaseModel):
    """
    X-Field Gaussian 模型

    使用 density（密度）替代 opacity（不透明度），
    采用 Softplus 激活函数确保密度非负。
    无球谐特征，输出为单通道灰度 X-ray 投影。
    """

    def setup_functions(self, scale_bound=None):
        """设置激活函数

        Args:
            scale_bound: 可选的尺度边界 [min, max]，用于约束密度尺度
        """

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if scale_bound is not None:
            scale_min_bound, scale_max_bound = scale_bound
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * scale_max_bound + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / scale_max_bound) + 1e-8
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.density_activation = torch.nn.Softplus()
        self.density_inverse_activation = inverse_softplus
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        super().__init__()

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions(scale_bound)

    def capture(self) -> Dict:
        """捕获模型状态"""
        return {
            'xyz': self._xyz,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'density': self._density,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer_state': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }

    def restore(self, state: Dict, opt):
        """从保存状态恢复模型"""
        self._xyz = state['xyz']
        self._scaling = state['scaling']
        self._rotation = state['rotation']
        self._density = state['density']
        self.max_radii2D = state['max_radii2D']
        self.xyz_gradient_accum = state['xyz_gradient_accum']
        self.denom = state['denom']
        self.spatial_lr_scale = state['spatial_lr_scale']
        self.training_setup(opt)
        self.optimizer.load_state_dict(state['optimizer_state'])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        """返回密度值（Softplus 激活后）"""
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        """从点云创建高斯

        Args:
            xyz: 坐标 [N, 3]
            density: 密度值 [N, 1]
            spatial_lr_scale: 空间学习率缩放
        """
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        print(f"Initialize X-Field Gaussians from {fused_point_cloud.shape[0]} points")

        if density is not None:
            fused_density = (
                self.density_inverse_activation(
                    torch.tensor(np.asarray(density)).float().cuda()
                )
            )
        else:
            fused_density = self.density_inverse_activation(
                torch.ones((fused_point_cloud.shape[0], 1), device="cuda") * 0.1
            )

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud),
            0.001 ** 2,
        )
        scales = self.scaling_inverse_activation(
            torch.sqrt(dist2)
        )[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_r2_init(self, init_path: str, spatial_lr_scale: float = 1.0):
        """从 R²-Gaussian 初始化文件创建高斯

        支持 .npy 和 .ply 格式：
        - .npy: [N, 3+] 点云坐标，第4列为密度
        - .ply: PLY 点云文件（通过 fetchPly）

        Args:
            init_path: 初始化文件路径
            spatial_lr_scale: 空间学习率缩放
        """
        if init_path.endswith('.npy'):
            data = np.load(init_path)
            xyz = data[:, :3]
            if data.shape[1] > 3:
                density = data[:, 3:4]
            else:
                density = np.ones((xyz.shape[0], 1), dtype=np.float32) * 0.1
            self.create_from_pcd(xyz, density, spatial_lr_scale)
        else:
            from r2_gaussian.utils.graphics_utils import fetchPly
            point_cloud = fetchPly(init_path)
            xyz = np.asarray(point_cloud.points)
            density = np.ones((xyz.shape[0], 1), dtype=np.float32) * 0.1
            self.create_from_pcd(xyz, density, spatial_lr_scale)

    def training_setup(self, training_args):
        """设置训练优化器

        Args:
            training_args: 优化参数对象，需包含以下属性：
                position_lr_init, position_lr_final, position_lr_max_steps,
                density_lr_init, density_lr_final, density_lr_max_steps,
                scaling_lr_init, scaling_lr_final, scaling_lr_max_steps,
                rotation_lr_init, rotation_lr_final, rotation_lr_max_steps
        """
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """更新学习率"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def save_ply(self, path):
        """保存为 PLY 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self._density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, densities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def load_ply(self, path):
        """从 PLY 文件加载"""
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]

        scale_names = [
            p.name for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties
            if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._density = nn.Parameter(
            torch.tensor(densities, dtype=torch.float, device="cuda")
            .requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda")
            .requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda")
            .requires_grad_(True)
        )

    # ==================== 密集化操作 ====================

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
    ):
        d = {
            "xyz": new_xyz,
            "density": new_density,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat(
            [self.max_radii2D, new_max_radii2D], dim=-1
        )

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        """基于梯度和大尺度的高斯分裂"""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device="cuda", dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_gap(self, grads, grad_threshold, densify_scale_threshold, N=2):
        """密度梯度感知（GAP）密集化

        基于密度场的梯度（而非位置梯度）进行密集化，
        在密度变化剧烈的区域新增高斯。
        """
        num_points = self._xyz.size(0)
        num_samples = min(5000, num_points)
        k = 500
        batch_size = 256

        indices = torch.randperm(num_points, device=self._xyz.device)[:num_samples]
        sampled_xyz = self._xyz[indices]
        sampled_densities = self.density_inverse_activation(
            self.get_density[indices]
        ).squeeze(-1)
        all_densities = self.density_inverse_activation(
            self.get_density
        ).squeeze(-1)

        gradients = torch.zeros(num_samples, device=self._xyz.device)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_xyz = sampled_xyz[start:end]
            dist_matrix = torch.cdist(batch_xyz, self._xyz)
            _, knn_indices = torch.topk(
                dist_matrix, k=k, dim=1, largest=False
            )
            neighbor_densities = all_densities[knn_indices]
            gradients[start:end] = torch.abs(
                sampled_densities[start:end]
                - neighbor_densities.mean(dim=1)
            )

        local_mask = gradients > grad_threshold
        selected_pts_mask = torch.zeros(
            num_points, device=self._xyz.device, dtype=torch.bool
        )
        selected_pts_mask[indices] = local_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device="cuda", dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        """基于梯度和小尺度的高斯克隆"""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_density

        self.densification_postfix(
            new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
        )

    def densify_and_prune(self, max_grad, min_density, max_screen_size,
                          max_scale, max_num_gaussians, densify_scale_threshold,
                          iteration, bbox=None):
        """完整的密集化与剪枝流程"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 密集化
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)
                if iteration > 1500:
                    self.densify_gap(grads, max_grad, densify_scale_threshold, N=2)

        # 剪枝低密度高斯
        prune_mask = (self.get_density < min_density).squeeze()

        # 剪枝超出边界的高斯
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )
            prune_mask = prune_mask | prune_mask_xyz

        # 剪枝大屏幕尺寸的高斯
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_points_vs

        if prune_mask.any():
            self.prune_points(prune_mask)

        # 重置密度
        torch.cuda.empty_cache()

    # ==================== GaussianBaseModel 接口实现 ====================

    def get_num_points(self) -> int:
        """返回高斯点数量"""
        return self.get_xyz.shape[0]

    def get_trainable_params(self) -> List[Dict]:
        """返回可训练参数组"""
        return [
            {"params": [self._xyz], "lr": self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._density], "name": "density"},
            {"params": [self._scaling], "name": "scaling"},
            {"params": [self._rotation], "name": "rotation"},
        ]

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """添加密集化统计信息"""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
