import os
import math

import torch
from pytorch3d.ops import knn_points

from scene.network import ODE_DisplacementModel
from utils.general_utils import get_expon_lr_func
from utils.rigid_utils import exp_se3
from utils.system_utils import searchForMaxIteration


class DeformModel:
    def __init__(self, encoder_config):
        self.OdeTransformer = ODE_DisplacementModel(encoder_config).cuda()

        self.optimizer = None
        self.network_lr_scale = 5.0
        self.net_lr_scheduler = None

    def _build_deform_pkg(self, feature, v, w, s, r, compute_div_loss=False, div_mode="patch"):
        x = feature[..., :3]
        theta = torch.norm(w, dim=-1, keepdim=True)
        w = w / (theta + 1e-5)
        v = v / (theta + 1e-5)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)
        d_xyz = self.se3_transform_displacement(transform, x)

        pkg = {
            "d_xyz": d_xyz,
            "d_rotation": r,
            "d_scaling": s,
        }

        if compute_div_loss:
            pkg["div_loss"] = self.divergence_loss(
                x.detach(), d_xyz, mode=div_mode, feature=feature.detach()
            )

        return pkg

    def step(self, feature, t, infer=False, compute_div_loss=False, div_mode="patch"):
        v, w, s, r = self.OdeTransformer(feature.unsqueeze(0), float(t), infer)

        return self._build_deform_pkg(
            feature,
            v,
            w,
            s,
            r,
            compute_div_loss=compute_div_loss,
            div_mode=div_mode,
        )

    def divergence_loss(
        self,
        xyz: torch.Tensor,
        d_xyz: torch.Tensor,
        mode: str = "sph",
        n_samples: int = 512,
        k_neighbors: int = 8,
        feature: torch.Tensor = None,
    ) -> torch.Tensor:
        if mode == "patch":
            return self._div_loss_patch(xyz, d_xyz, n_samples)
        if mode == "point":
            return self._div_loss_point(xyz, d_xyz, n_samples, k_neighbors)
        if mode == "sph":
            return self._div_loss_sph(xyz, d_xyz, feature, n_samples, k_neighbors)
        raise ValueError(f"div mode must be 'patch', 'point', or 'sph', got '{mode}'")

    def _div_loss_patch(
        self, xyz: torch.Tensor, d_xyz: torch.Tensor, n_patches: int
    ) -> torch.Tensor:
        knn_idx = self.OdeTransformer.encoder.cache_knn_idx
        if knn_idx is None:
            return d_xyz.new_zeros(1).squeeze()

        knn_idx = knn_idx[0]
        G, K = knn_idx.shape
        if K < 2:
            return d_xyz.new_zeros(1).squeeze()

        N = xyz.shape[0]
        if int(knn_idx.max()) >= N:
            self.OdeTransformer.encoder.cache_knn_idx = None
            return d_xyz.new_zeros(1).squeeze()

        M = min(n_patches, G)
        perm = torch.randperm(G, device=xyz.device)[:M]
        patch_idx = knn_idx[perm]

        seed_idx = patch_idx[:, 0]
        nbr_idx = patch_idx[:, 1:]

        x_seeds = xyz[seed_idx]
        x_nbrs = xyz[nbr_idx]
        v_seeds = d_xyz[seed_idx]
        v_nbrs = d_xyz[nbr_idx]

        dx = x_nbrs - x_seeds.unsqueeze(1)
        dv = v_nbrs - v_seeds.unsqueeze(1)
        dist2 = (dx * dx).sum(-1, keepdim=True).clamp(min=1e-8)
        div_i = ((dv * dx) / dist2).sum(-1).mean(-1)

        return (div_i**2).mean()

    def _div_loss_point(
        self, xyz: torch.Tensor, d_xyz: torch.Tensor, n_samples: int, k_neighbors: int
    ) -> torch.Tensor:
        N = xyz.shape[0]
        M = min(n_samples, N)
        perm = torch.randperm(N, device=xyz.device)[:M]

        x_seeds = xyz[perm]
        v_seeds = d_xyz[perm]

        with torch.no_grad():
            knn_result = knn_points(
                x_seeds.unsqueeze(0),
                xyz.unsqueeze(0),
                K=k_neighbors + 1,
                return_sorted=False,
            )
        knn_idx = knn_result.idx[0, :, 1:]

        x_nbrs = xyz[knn_idx]
        v_nbrs = d_xyz[knn_idx]

        dx = x_nbrs - x_seeds.unsqueeze(1)
        dv = v_nbrs - v_seeds.unsqueeze(1)
        dist2 = (dx * dx).sum(-1, keepdim=True).clamp(min=1e-8)
        div_i = ((dv * dx) / dist2).sum(-1).mean(-1)

        return (div_i**2).mean()

    def _sph_kernel_grad(
        self, dx: torch.Tensor, h: torch.Tensor, kernel: str = "gaussian"
    ) -> torch.Tensor:
        if kernel == "gaussian":

            r2 = (dx * dx).sum(-1, keepdim=True).clamp(min=1e-12)
            h2 = (h * h).clamp(min=1e-12)
            W = torch.exp(-r2 / h2)
            return (-2.0 / h2) * dx * W

        if kernel == "cubic":

            r = (dx * dx).sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
            q = (r / h).clamp(max=2.0)
            sigma = 8.0 / (math.pi * h**3)

            dWdq = torch.zeros_like(r)
            m1 = q < 1.0
            m2 = (q >= 1.0) & (q < 2.0)
            dWdq = torch.where(m1, sigma * q * (-3.0 + 2.25 * q), dWdq)
            dWdq = torch.where(m2, sigma * (-0.75) * (2.0 - q) ** 2, dWdq)

            return (dWdq / h) * (dx / r.clamp(min=1e-8))

        raise ValueError(f"Unknown SPH kernel: '{kernel}'. Use 'gaussian' or 'cubic'.")

    def _div_loss_sph(
        self,
        xyz: torch.Tensor,
        d_xyz: torch.Tensor,
        feature: torch.Tensor,
        n_samples: int,
        k_neighbors: int,
        kernel: str = "gaussian",
        h_scale: float = 1.0,
    ) -> torch.Tensor:

        knn_idx = self.OdeTransformer.encoder.cache_knn_idx

        if knn_idx is not None:
            knn_idx = knn_idx[0]
            G = knn_idx.shape[0]
            M = min(n_samples, G)
            perm = torch.randperm(G, device=xyz.device)[:M]
            patch_idx = knn_idx[perm]
            seed_idx = patch_idx[:, 0]
            nbr_idx = patch_idx[:, 1:]
        else:
            N = xyz.shape[0]
            M = min(n_samples, N)
            perm = torch.randperm(N, device=xyz.device)[:M]
            seed_idx = perm
            with torch.no_grad():
                knn_result = knn_points(
                    xyz[perm].unsqueeze(0),
                    xyz.unsqueeze(0),
                    K=k_neighbors + 1,
                    return_sorted=False,
                )
            nbr_idx = knn_result.idx[0, :, 1:]

        K_nbr = nbr_idx.shape[-1]
        if K_nbr < 1:
            return d_xyz.new_zeros(1).squeeze()

        x_i = xyz[seed_idx]
        x_j = xyz[nbr_idx]
        v_i = d_xyz[seed_idx]
        v_j = d_xyz[nbr_idx]

        dx = x_i.unsqueeze(1) - x_j

        r = dx.norm(dim=-1)
        h = (h_scale * r.mean(dim=-1)).clamp(min=1e-6)
        h = h.view(M, 1, 1)

        if feature is not None:
            scale_j = feature[nbr_idx, 10:13]
            alpha_j = feature[nbr_idx, 13:14]
            vol_j = (alpha_j * scale_j.prod(dim=-1, keepdim=True)).clamp(min=1e-8)
        else:
            vol_j = torch.ones(M, K_nbr, 1, device=xyz.device)

        vol_j = vol_j / (vol_j.sum(dim=1, keepdim=True) + 1e-8)

        grad_W = self._sph_kernel_grad(dx, h, kernel=kernel)

        dv = v_j - v_i.unsqueeze(1)
        div_i = (vol_j * (dv * grad_W).sum(dim=-1, keepdim=True)).sum(dim=1).squeeze(-1)

        return (div_i**2).mean()

    def train_setting(self, training_args):

        self.network_lr_scale = training_args.network_lr_scale

        l = [
            {
                "params": list(self.OdeTransformer.parameters()),
                "lr": training_args.position_lr_init * self.network_lr_scale,
                "name": "ode",
            }
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.net_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.network_lr_scale,
            lr_final=training_args.position_lr_final * self.network_lr_scale * 0.5,
            lr_delay_mult=training_args.position_lr_delay_mult * self.network_lr_scale,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.OdeTransformer.state_dict(), os.path.join(out_weights_path, "deform.pth"))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "deform"))
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(iteration))

        print("Load weight:", weights_path)
        ode_weight = torch.load(weights_path, map_location="cuda")
        self.OdeTransformer.load_state_dict(ode_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "ode":
                lr = self.net_lr_scheduler(iteration)
                param_group["lr"] = lr

    def se3_transform_displacement(self, T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_de = x.detach()
        R = T[:, :3, :3]
        p = T[:, :3, 3]
        x_rot = torch.bmm(R, x_de.unsqueeze(-1)).squeeze(-1)
        x_new = x_rot + p
        displacement = x_new - x_de

        return displacement
