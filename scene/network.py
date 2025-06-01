import os
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchdiffeq import odeint

from hashencoder.hashgrid import HashEncoder
from utils.engine_utils import particle_position_tensor_to_ply
from typing import *

from torch.nn import Module
from torch import Tensor


class SpaceEncoder(nn.Module):
    def __init__(
            self,
            canonical_num_levels=16,
            canonical_level_dim=2,
            canonical_base_resolution=16,
            canonical_desired_resolution=2048,
            canonical_log2_hashmap_size=19,

            deform_num_levels=32,
            deform_level_dim=2,

            bound=1.6,
    ):
        super(SpaceEncoder, self).__init__()
        self.out_dim = canonical_num_levels * canonical_level_dim + deform_num_levels * deform_level_dim * 3
        self.canonical_num_levels = canonical_num_levels
        self.canonical_level_dim = canonical_level_dim
        self.deform_num_levels = deform_num_levels
        self.deform_level_dim = deform_level_dim
        self.bound = bound

        self.xyz_encoding = HashEncoder(
            input_dim=3,
            num_levels=canonical_num_levels,
            level_dim=canonical_level_dim,
            per_level_scale=2,
            base_resolution=canonical_base_resolution,
            log2_hashmap_size=canonical_log2_hashmap_size,
            desired_resolution=canonical_desired_resolution,
        )

    def forward(self, xyz):
        return self.xyz_encoding(xyz, size=self.bound)


def get_pos_emb(num_groups: int, hidden_size: int) -> Tensor:
    '''
    Get absolute positional embedding
    input:
        num_groups: int, the number of groups, i.e., the number of positions
        hidden_size: int, the dimension of the embedding
    output:
        position: Tensor, (G, H), the positional embedding
    '''
    assert hidden_size % 2 == 0, 'The hidden size must be even.'
    pos_emb = torch.zeros(num_groups, hidden_size)
    position = torch.arange(num_groups).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    return pos_emb


class Grouper(Module):
    '''
    Grouping the points based on the centers with FPS and KNN
    input:
        x: Tensor, (B, N, 3), the coordinates of the points
    output:
        neighbors: Tensor, (B, G, M, F), the features of the neighbors
        centers: Tensor, (B, G, 3), the coordinates of the centersS
    '''

    def __init__(
            self,
            num_groups: int,
            group_size: int,
            export_path: Optional[str] = None,
    ) -> None:
        super(Grouper, self).__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.export_path = export_path

    def forward(self, x: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        # 1. Select the centers with FPS
        centers = fps(x, self.num_groups)  # (B, N, 3) -> (B, G, 3)
        # 2. Select the neighbors with KNN
        neighbors, nearest_center_idx = knn(x, centers, features,
                                            self.group_size)  # (B, N, 3), (B, G, 3) -> (B, G, M, 3)
        if self.export_path:
            # save fps centers to .ply file for visualization
            particle_position_tensor_to_ply(centers.squeeze(0), os.path.join(self.export_path, 'fps_centers.ply'))
            print(f'FPS centers saved to {os.path.join(self.export_path, "fps_centers.ply")}')
        return neighbors, nearest_center_idx


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    fps_idx = centroids.squeeze(0)  # (1, npoint) -> (npoint)
    fps_data = xyz[0, fps_idx].unsqueeze(0)  # (1, N, 3) -> (1, npoint, 3)
    return fps_data


def knn(x: Tensor, centers: Tensor, features: Tensor, group_size: int) -> Tensor:
    '''
    Select the neighbors with KNN
    input:
        x: Tensor, (B, N, 3), the coordinates of the points
        centers: Tensor, (B, G, 3), the coordinates of the centers
        group_size: int, the number of neighbors
    output:
        neighbors: Tensor, (B, G, M, F), the features of the neighbors
    '''

    distances = square_distance(centers, x)  # (B, G, 3), (B, N, 3) -> (B, G, N)
    _, neighbors_idx = torch.topk(distances, group_size, dim=-1, largest=False, sorted=False)  # (B, G, N) -> (B, G, M)
    # don't need idx_base since batch size is 1. TODO: add batch size support
    neighbors_idx = neighbors_idx.flatten()  # (B, G, M) -> (B * G * M)
    neighbors = features.reshape(-1, features.shape[-1])[neighbors_idx]  # (B * N, F) -> (B * G * M, F)
    neighbors = neighbors.reshape(1, -1, group_size, features.shape[-1])  # (B * G * M, F) -> (B, G, M, F)

    nearest_center_idx = torch.argmin(distances.permute(0, 2, 1), dim=-1)  # (B, N, G) -> (B, N)

    return neighbors, nearest_center_idx


def square_distance(src, dst):
    '''
    Calculate the square distance between the src and dst
    input:
        src: Tensor, (B, N, 3), the coordinates of the source points
        dst: Tensor, (B, M, 3), the coordinates of the destination points
    output:
        dist: Tensor, (B, N, M), the square distance between the src and dst
    '''
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(src.shape[0], src.shape[1], 1)
    dist += torch.sum(dst ** 2, -1).view(dst.shape[0], 1, dst.shape[1])
    return dist


class GroupEncoder(Module):
    '''
    Encode the groups
    input:
        groups: Tensor, (B, G, M, F), the coordinates of the neighbors of the centers
    output:
        features: Tensor, (B, G, H), the features of the centers
    '''

    def __init__(
            self,
            feat_dim: int = 3,
            hidden_dim: int = 128,
            out_dim: int = 128,
    ) -> None:
        super(GroupEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 4, out_dim, 1),
        )

    def forward(self, group_features: Tensor) -> Tensor:
        B, G, M, F = group_features.shape
        group_features = group_features.reshape(B * G, F, M)  # (B, G, M, F) -> (B * G, F, M)
        group_features = self.conv1(group_features)  # (B * G, F, M) -> (B * G, 2 * H, M)
        group_features_global_max = torch.max(group_features, dim=-1, keepdim=True)[0]  # (B * G, 2 * H, M) -> (B * G, 2 * H, 1)
        group_features_global_max = group_features_global_max.expand(-1, -1,
                                                                     M)  # (B * G, 2 * H, 1) -> (B * G, 2 * H, M)
        group_features = torch.cat([group_features, group_features_global_max],
                                   dim=1)  # (B * G, 2 * H, M) -> (B * G, 4 * H, M)
        group_features = self.conv2(group_features)  # (B * G, 4 * H, M) -> (B * G, O, M)
        group_features = torch.max(group_features, dim=-1)[0]  # (B * G, O, M) -> (B * G, O)
        group_features = group_features.reshape(B, G, -1)  # (B * G, O) -> (B, G, O)
        return group_features


class Attention(Module):
    '''
    Attention mechanism
    input:
        x: Tensor, (B, N, F), the features of the points
    output:
        x: Tensor, (B, N, F), the updated features of the points
    '''

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
    ) -> None:
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, F = x.shape
        qkv = self.qkv(x)  # (B, N, F) -> (B, N, 3 * F)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (B, N, 3 * F) -> (B, N, 3, H, D) -> (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (3, B, H, N, D) -> (B, H, N, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, D), (B, H, D, N) -> (B, H, N, N)
        attn = attn.softmax(dim=-1)  # (B, H, N, N)
        attn = self.dropout(attn)  # (B, H, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # (B, H, N, N), (B, H, N, D) -> (B, N, H * D) = (B, N, F)
        x = self.fc(x)  # (B, N, H * D) -> (B, N, F)
        return x


class Block(Module):
    '''
    Transformer block
    input:
        x: Tensor, (B, N, F), the features of the points
    output:
        x: Tensor, (B, N, F), the updated features of the points
    '''

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
    ) -> None:
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))  # (B, N, F)
        x = x + self.mlp(self.norm2(x))  # (B, N, F)
        return x


class KNNTransformerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            hidden_size: int = 128,
            num_groups: int = 2048,
            group_size: int = 32,
            depth: int = 16,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.grouper = Grouper(num_groups, group_size)
        self.group_encoder = GroupEncoder(in_channels, hidden_dim=hidden_size // 4, out_dim=hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_groups, hidden_size))
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.cache = None

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, y: Tensor):
        # x: (B, N, 3+F), t: (B, 1)
        x = y[..., :3]
        features = y
        # 1. KNNTransformer
        if self.cache is None:
            with torch.no_grad():
                neighbors, nearest_center_idx = self.grouper(x, features)  # (1, N, 3), (1, N, F) -> (1, G, M, F), (1, N)
            neighbors = neighbors.detach()  # avoid backpropagate to the FPS and KNN
            nearest_center_idx = nearest_center_idx.detach()
            self.cache = (neighbors, nearest_center_idx)
        else:
            neighbors, nearest_center_idx = self.cache

        # 2.pos+att
        encoded_features = self.group_encoder(neighbors)    # (B, G, M, F) -> (B, G, H)
        encoded_features += self.pos_emb
        for block in self.blocks:
            encoded_features = block(encoded_features)

        # 3. maxpool
        global_feature = encoded_features.max(dim=1)[0]  # (B, G, H) → (B, H)
        z0 = self.to_latent(global_feature)  # (B, H) → (B, hidden_size)
        return z0


class ODEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, t: float, z: Tensor) -> Tensor:
        t_tensor = torch.ones(z.size(0), 1, device=z.device) * t
        dzdt = self.net(torch.cat([z, t_tensor], dim=1))
        return dzdt


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int = 9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim))

    def forward(self, z_t: Tensor, points: Tensor) -> Tensor:
        # z_t: (B, D), points: (B, N, 32)
        B, N, _ = points.shape
        z_t = z_t.unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([z_t, points], dim=2)
        displacement = self.mlp(h)
        return displacement


class RSDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int = 9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim))

    def forward(self, z_t: Tensor, points: Tensor) -> Tensor:
        # z_t: (B, D), points: (B, N, 32)
        B, N, _ = points.shape
        z_t = z_t.unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([z_t, points], dim=2)
        displacement = self.mlp(h)
        return displacement


class ODE_DisplacementModel(nn.Module):
    def __init__(self, encoder_config, ode_hidden=256):
        super().__init__()
        self.encoder = KNNTransformerEncoder(**encoder_config)
        self.ode_func = ODEFunc(input_dim=encoder_config["hidden_size"], hidden_dim=ode_hidden)
        self.decoder = Decoder(latent_dim=encoder_config["hidden_size"], output_dim=19)
        self.rsdecoder = RSDecoder(latent_dim=encoder_config["hidden_size"], output_dim=7)
        self.zt = None

        self.epo = torch.tensor([0.001], device='cuda')

    def forward(self, x: Tensor, t: Tensor):
        # x: (B, N, 3 + 32), t: (B, 1)
        # 1. encoder: latent state z0
        z0 = self.encoder(x)

        # 2. NODE: z0 -> z_t
        ts = torch.tensor([0.0, torch.max(t.squeeze(), self.epo)], device=x.device)
        z_t = odeint(self.ode_func, z0, ts, method='rk4')
        self.zt = z_t
        z_t = z_t[-1]

        # 3. decoder
        mot = self.decoder(z_t, x[..., 3:])
        rs = self.rsdecoder(z_t, x[..., 3:])

        return mot, rs
