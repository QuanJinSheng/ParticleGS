import random

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, jacrev
from torchdiffeq import odeint_adjoint as odeint

from scene.network import ODE_DisplacementModel, SpaceEncoder
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(
            self, 
            grid_args,
            encoder_config,
            scale_xyz=1.0,
            sh_enable=False
        ):
        self.HashEncoder = SpaceEncoder(**grid_args).cuda()
        self.OdeTransformer = ODE_DisplacementModel(encoder_config, ode_hidden=256).cuda()

        self.optimizer = None
        self.network_lr_scale = 5.0
        self.grid_lr_scale = 100.0

        if type(scale_xyz) is float:
                scale_xyz = [scale_xyz for _ in range(3)]
        else:
            assert len(scale_xyz) == 3
        self.scale_xyz = torch.tensor(scale_xyz, device="cuda", dtype=torch.float32)

        self.hash_lr_scheduler = None
        self.net_lr_scheduler = None

        self.theta_epo = 1e-7
        self.epsilon_v = 1e-6
        self.v_epo = 1e-4
        self.spatial_h_cache = None
        self.sh_enable = sh_enable
        
    def step(self, xyz, t, fixed_attention=False):
        xyz = xyz * self.scale_xyz[None, ...]
        xyz.requires_grad_(True)

        # get feature
        if fixed_attention and self.spatial_h_cache is not None:
            spatial_h = self.spatial_h_cache
        else:
            spatial_h = self.HashEncoder(xyz)
            self.spatial_h_cache = spatial_h

        x = torch.cat([xyz, spatial_h], dim=-1).unsqueeze(0)
        mot, rs = self.OdeTransformer(x, torch.tensor([[t]], dtype=torch.float32, device='cuda'))
        rs = rs.squeeze(0)
        p, v, w, kv, kw, s, h = mot[..., :3], mot[..., 3:6], mot[..., 6:9], mot[..., 9:10], mot[..., 10:11], mot[..., 11:14], mot[..., 14:]
        d_xyz = self.compute_displacement_batch(xyz.unsqueeze(0), p, v, w, kv, kw, s, h, self.sh_enable).squeeze(0)
        rotation, scaling= rs[..., :4], rs[..., 4:]

        return {
            "d_xyz": d_xyz,
            "d_rotation": rotation, 
            "d_scaling": scaling,
            "mot": mot
        }
    
    def train_setting(self, training_args):
        self.grid_lr_scale = training_args.grid_lr_scale
        self.network_lr_scale = training_args.network_lr_scale

        l = [
            {'params': list(self.HashEncoder.parameters()),
             'lr': training_args.position_lr_init * self.grid_lr_scale,
             "name": "hash"},
            {'params': list(self.OdeTransformer.parameters()),
             'lr': training_args.position_lr_init * self.network_lr_scale,
             "name": "ode"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.hash_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.grid_lr_scale,
                                                   lr_final=training_args.position_lr_final * self.grid_lr_scale,
                                                   lr_delay_mult=training_args.position_lr_delay_mult,
                                                   max_steps=training_args.deform_lr_max_steps)
        self.net_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.network_lr_scale,
                                                  lr_final=training_args.position_lr_final * self.network_lr_scale * 0.5,
                                                  lr_delay_mult=training_args.position_lr_delay_mult * self.network_lr_scale,
                                                  max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save((self.HashEncoder.state_dict(), self.OdeTransformer.state_dict()), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        hash_weight, ode_weight = torch.load(weights_path, map_location='cuda')
        self.HashEncoder.load_state_dict(hash_weight)
        self.OdeTransformer.load_state_dict(ode_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "hash":
                lr = self.hash_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == 'ode':
                lr = self.net_lr_scheduler(iteration)
                param_group['lr'] = lr

    def skew_batch(self, w):
        """
        Skew-symmetric matrix generation
        Args:
            w: Angular velocity vector (B, N, 3)
        Returns:
            K: Skew-symmetric matrix (B, N, 3, 3)
        """

        B, N, _ = w.shape
        w_flat = w.view(-1, 3)  # (B*N, 3)

        zeros = torch.zeros(B * N, 1, device=w.device)  # (B*N,1)

        row1 = torch.cat([zeros, -w_flat[:, 2:], -w_flat[:, 1:2]], dim=1)
        row2 = torch.cat([w_flat[:, 2:], zeros, -w_flat[:, 0:1]], dim=1)
        row3 = torch.cat([-w_flat[:, 1:2], w_flat[:, 0:1], zeros], dim=1)
        K_flat = torch.stack([row1, row2, row3], dim=1)  # (B*N, 3, 3)

        return K_flat.view(B, N, 3, 3)  # (B, N, 3, 3)

    def compute_displacement_batch(self, x, p, v, w, kv, kw, s, h, sh_enable):
        """
        Args:
            x: Original position (B, N, 3)
            p: Rotation origin (B, N, 3)
            v: Linear velocity direction (B, N, 3)
            w: Angular velocity direction (B, N, 3)
            kv: Linear velocity magnitude (B, N, 1)
            kw: Angular velocity magnitude (B, N, 1)
            s: Scaling factors (B, N, 3)
            h: Shearing factors (B, N, 3)
            sh_enable: Whether scaling and shearing are enabled

        Returns:
            dx: Displacement x1 - x (B, N, 3)
        """

        device = x.device
        B, N, _ = x.shape

        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        w = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-8)
        v_actual = v * kv
        w_actual = w * kw

        theta = torch.norm(w_actual, dim=2, keepdim=True) + 1e-8
        k = w_actual / theta

        # Rodrigues
        K = self.skew_batch(k)
        I = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3)
        theta_flat = theta.view(B * N, 1, 1)
        K_flat = K.view(B * N, 3, 3)

        R_flat = I.view(B * N, 3, 3) + \
                 torch.sin(theta_flat) * K_flat + \
                 (1 - torch.cos(theta_flat)) * (K_flat @ K_flat)
        R = R_flat.view(B, N, 3, 3)

        if sh_enable:
            S = torch.zeros(B, N, 3, 3, device=device)
            S[:, :, 0, 0] = s[:, :, 0]
            S[:, :, 1, 1] = s[:, :, 1]
            S[:, :, 2, 2] = s[:, :, 2]

            H = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3).clone()
            H[:, :, 0, 1] = h[:, :, 0]
            H[:, :, 0, 2] = h[:, :, 1]
            H[:, :, 1, 2] = h[:, :, 2]

            A = torch.einsum('bnij,bnjk,bnkl->bnil', S, R, H)
        else:
            A = R

        offset = x - p
        transformed = torch.einsum('bnij,bnj->bni', A, offset)
        x1 = transformed + p + v_actual

        return x1 - x

