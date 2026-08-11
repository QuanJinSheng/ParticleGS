import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from typing import Optional, Tuple

from torch.nn import Module
from torch import Tensor
from pytorch3d.ops import sample_farthest_points, knn_points


def stratified_fps(x: Tensor, displacements: Tensor, num_groups: int,
                   ratio_high: float = 0.1, sample_ratio_high: float = 0.8) -> Tensor:
    '''
    Stratified motion-aware FPS: over-sample high-displacement points.
    - Top ratio_high of points (by displacement magnitude) are "high-motion"
    - sample_ratio_high fraction of num_groups is drawn from the high-motion stratum
    Always returns exactly num_groups centers; if N < num_groups, falls back to uniform FPS.
    input:
        x: (B, N, 3)
        displacements: (B, N, 3) or (B, N) displacement magnitudes
        num_groups: total number of centers to select
        ratio_high: fraction of points classified as high-motion
        sample_ratio_high: fraction of centers sampled from the high-motion stratum
    output:
        centers: (B, num_groups, 3)
    '''
    B, N, _ = x.shape

    # Fallback: too few points to stratify
    if N <= num_groups:
        centers, _ = sample_farthest_points(x, K=min(num_groups, N))
        return centers

    if displacements.dim() == 3:
        disp_norms = displacements.norm(dim=-1)  # (B, N)
    else:
        disp_norms = displacements  # (B, N)

    # Fallback: displacement size mismatch (e.g. after densification)
    if disp_norms.shape[1] != N:
        centers, _ = sample_farthest_points(x, K=num_groups)
        return centers

    n_high = max(1, int(N * ratio_high))
    n_high = min(n_high, N - 1)
    n_low  = N - n_high

    # Desired samples per stratum
    k_high = min(int(num_groups * sample_ratio_high), n_high)
    k_low  = num_groups - k_high   # always compensate so total = num_groups
    # If low stratum can't cover k_low, push surplus back to high
    if k_low > n_low:
        k_low  = n_low
        k_high = num_groups - k_low
        k_high = min(k_high, n_high)

    # Sort by displacement descending
    sorted_idx = torch.argsort(disp_norms, dim=-1, descending=True)  # (B, N)
    high_idx = sorted_idx[:, :n_high]
    low_idx  = sorted_idx[:, n_high:]

    def gather_pts(pts, idx):
        return pts.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

    x_high = gather_pts(x, high_idx)  # (B, n_high, 3)
    x_low  = gather_pts(x, low_idx)   # (B, n_low, 3)

    centers_high, _ = sample_farthest_points(x_high, K=k_high)
    centers_low,  _ = sample_farthest_points(x_low,  K=k_low)

    centers = torch.cat([centers_high, centers_low], dim=1)  # (B, num_groups, 3)
    return centers


def displacement_weighted_fps(x: Tensor, displacements: Tensor, num_groups: int,
                              alpha: float = 1.0) -> Tensor:
    '''
    Weighted FPS: selection score = spatial_distance * (1 + alpha * norm_weight),
    so high-displacement points compete more aggressively to be chosen as centers.
    Starts from the highest-displacement point to seed the selection.
    input:
        x: (B, N, 3)
        displacements: (B, N, 3) or (B, N)
        num_groups: number of centers
        alpha: weight strength (0 = uniform FPS, larger = stronger motion bias)
    output:
        centers: (B, num_groups, 3)
    '''
    B, N, _ = x.shape
    device = x.device

    if displacements.dim() == 3:
        disp_norms = displacements.norm(dim=-1)  # (B, N)
    else:
        disp_norms = displacements

    # Normalize weights to [0, 1]
    w_min = disp_norms.min(dim=1, keepdim=True)[0]
    w_max = disp_norms.max(dim=1, keepdim=True)[0]
    weights = (disp_norms - w_min) / (w_max - w_min + 1e-8)  # (B, N)

    selected = torch.zeros(B, num_groups, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    # Seed: highest-displacement point
    selected[:, 0] = weights.argmax(dim=1)
    last_pts = x[batch_idx, selected[:, 0]]              # (B, 3)
    min_dists = ((x - last_pts.unsqueeze(1)) ** 2).sum(-1)  # (B, N)

    for i in range(1, num_groups):
        score = min_dists * (1.0 + alpha * weights)      # (B, N)
        next_idx = score.argmax(dim=1)                   # (B,)
        selected[:, i] = next_idx
        next_pts = x[batch_idx, next_idx]                # (B, 3)
        new_dists = ((x - next_pts.unsqueeze(1)) ** 2).sum(-1)
        min_dists = torch.min(min_dists, new_dists)

    idx_exp = selected.unsqueeze(-1).expand(-1, -1, 3)
    centers = x.gather(1, idx_exp)                       # (B, num_groups, 3)
    return centers


def knn(x: Tensor, centers: Tensor, features: Tensor, group_size: int) -> Tuple[Tensor, Tensor]:
    '''
    Select the neighbors with KNN
    input:
        x: Tensor, (B, N, 3), the coordinates of the points
        centers: Tensor, (B, G, 3), the coordinates of the centers
        features: Tensor, (B, N, F), the features of the points
        group_size: int, the number of neighbors
    output:
        neighbors: Tensor, (B, G, K, F), the features of the neighbors
        neighbors_idx: Tensor, (B, G, K), indices into x of each neighbor
    '''
    B, N, _ = x.shape

    knn_result = knn_points(
        p1=centers,
        p2=x,
        K=group_size,
        norm=2,
        return_nn=False,
        return_sorted=False,
    )
    neighbors_idx = knn_result.idx  # (B, G, K)

    neighbors_idx_flat = neighbors_idx.reshape(-1)
    features_flat = features.reshape(-1, features.shape[-1])
    neighbors_flat = features_flat[neighbors_idx_flat]
    neighbors = neighbors_flat.reshape(B, neighbors_idx.shape[1], group_size, features.shape[-1])

    return neighbors, neighbors_idx

class Grouper(Module):
    '''
    motion_mode:
      None        - standard uniform FPS (default)
      'stratified'- stratified FPS: over-sample high-displacement points
      'weighted'  - displacement-weighted FPS (slower, more continuous bias)
    motion_kwargs: extra keyword arguments forwarded to the chosen sampling fn
      stratified: ratio_high (0.1), sample_ratio_high (0.8)
      weighted:   alpha (1.0)
    '''
    def __init__(self, num_groups: int, group_size: int,
                 motion_mode: Optional[str] = None,
                 motion_kwargs: Optional[dict] = None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        assert motion_mode in (None, 'stratified', 'weighted'), \
            f"motion_mode must be None, 'stratified', or 'weighted', got {motion_mode}"
        self.motion_mode = motion_mode
        self.motion_kwargs = motion_kwargs or {}
        # FPS visualization: set vis=True externally to record last centers
        self.vis: bool = False
        self.last_centers: Optional[Tensor] = None       # (G, 3) cpu float32
        self.last_high_idx: Optional[Tensor] = None      # (n_high,) indices of high-motion pts

    def forward(self, x: Tensor, features: Tensor,
                displacements: Optional[Tensor] = None) -> Tensor:
        '''
        x: (B, N, 3)  positions
        features: (B, N, F)
        displacements: (B, N, 3) or (B, N) optional; required when motion_mode is set
        '''
        if self.motion_mode is not None and displacements is not None:
            if self.motion_mode == 'stratified':
                centers = stratified_fps(x, displacements, self.num_groups,
                                         **self.motion_kwargs)
            else:  # 'weighted'
                centers = displacement_weighted_fps(x, displacements, self.num_groups,
                                                    **self.motion_kwargs)
        else:
            centers, _ = sample_farthest_points(x, K=self.num_groups)

        # Record for visualization if requested
        if self.vis:
            self.last_centers = centers[0].detach().cpu().float()  # (G, 3)
            if displacements is not None and displacements.shape[1] == x.shape[1]:
                disp_norms = displacements.norm(dim=-1) if displacements.dim() == 3 \
                    else displacements  # (1, N)
                n_high = max(1, int(x.shape[1] * self.motion_kwargs.get('ratio_high', 0.1)))
                sorted_idx = torch.argsort(disp_norms[0], descending=True)
                self.last_high_idx = sorted_idx[:n_high].detach().cpu()  # (n_high,)
            else:
                self.last_high_idx = None

        neighbors, neighbors_idx = knn(x, centers, features, self.group_size)
        return neighbors, neighbors_idx

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
        group_features_global_max = group_features_global_max.expand(-1, -1, M)  # (B * G, 2 * H, 1) -> (B * G, 2 * H, M)
        group_features = torch.cat([group_features, group_features_global_max], dim=1)  # (B * G, 2 * H, M) -> (B * G, 4 * H, M)
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

class SelfBlock(Module):
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
        super(SelfBlock, self).__init__()
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

class CrossAttentionAggregator(nn.Module):
    def __init__(self, dim, att_num_heads=8, k=4, learnable_queries=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = att_num_heads
        self.k = k
        self.head_dim = dim // att_num_heads
        assert dim % att_num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        if learnable_queries:
            self.queries = nn.Parameter(torch.randn(1, k, dim))
        else:
            self.queries = None

        self.norm_q = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, queries=None):
        B, N, H = x.shape

        if queries is None:
            assert self.queries is not None
            q = self.queries.expand(B, -1, -1)  # (B, K, H)
        else:
            q = queries  # (B, K, H)

        q = self.norm_q(q)
        x = self.norm_x(x)

        Q = self.q_proj(q).view(B, self.k, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,K,d)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # (B,h,N,d)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # (B,h,N,d)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,h,K,N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B,h,K,d)
        out = out.transpose(1, 2).contiguous().view(B, self.k, self.dim)  # (B,K,H)

        out = self.out_proj(out)
        out = q + self.dropout(out)

        return out  # (B,K,H)


class CrossBlock(nn.Module):
    '''
    Cross Attention block
    input:
        x: Tensor, (B, N, F), encoder features (K/V source)
    output:
        q: Tensor, (B, K, F), updated queries
    '''
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            q: int = 4,
            dropout: float = 0.0,
    ) -> None:
        super(CrossBlock, self).__init__()
        self.attn = CrossAttentionAggregator(dim, num_heads, q, learnable_queries=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, queries: torch.Tensor = None) -> torch.Tensor:
        # Cross-Attn
        q = self.attn(x, queries)  # (B, K, F)
        # FFN + Residual
        q = q + self.mlp(self.norm(q))
        return q  # (B, K, F)


class KNNTransformerEncoder(nn.Module):
    def __init__(
            self,
            in_dim: int = 14,
            hidden_size: int = 128,
            num_groups: int = 2048,
            group_size: int = 32,
            query: int = 4,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            l_dim: int = 32,
            **kwargs,           # absorb deprecated keys (e.g. query_dim) without crashing
    ) -> None:
        super().__init__()

        # query_dim is NOT a free parameter: it must equal l_dim so that
        #   z_t = l @ g_t.T = (N, l_dim) @ (l_dim, query) = (N, query)
        # decoder1 then uses in_dim=query.  query and l_dim may differ freely.
        self.query = query
        self.query_dim = l_dim   # derived from l_dim, not a free config param
        self.l_dim = l_dim
        self.hidden_dim = hidden_size
        self.grouper = Grouper(num_groups, group_size,
                               motion_mode='stratified',
                               motion_kwargs={'ratio_high': 0.1, 'sample_ratio_high': 0.6})
        self.group_encoder = GroupEncoder(self.l_dim, hidden_dim=hidden_size // 4, out_dim=hidden_size)
        self.g_pos_emb = nn.Parameter(torch.zeros(1, num_groups, hidden_size))
        self.cache = None          # cached neighbor features (B, G, K, F)
        self.cache_knn_idx = None  # cached neighbor indices (B, G, K) — reused for div loss
        self.g0_cache = None       # cached g0 (optional: refreshed every g0_refresh_interval steps)
        self._step_count = 0
        self.g0_refresh_interval = 1  # amortize transformer cost: recompute g0 every 4 steps; grad detached on skipped steps

        self.g_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.query_dim)
        )
        self.l_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.l_dim)
        )
        self.blocks = nn.ModuleList([CrossBlock(hidden_size, num_heads, q=num_groups // 2),
                                     SelfBlock(hidden_size, num_heads, mlp_ratio),
                                     CrossBlock(hidden_size, num_heads, q=num_groups // 8),
                                     SelfBlock(hidden_size, num_heads, mlp_ratio),
                                     CrossBlock(hidden_size, num_heads, q=num_groups // 32),
                                     SelfBlock(hidden_size, num_heads, mlp_ratio),
                                     CrossBlock(hidden_size, num_heads, q=self.query),
                                     SelfBlock(hidden_size, num_heads, mlp_ratio)
                                     ])

    def forward(self, features: Tensor, displacements: Optional[Tensor] = None):
        '''
        features: (B, N, 14)
        displacements: (B, N, 3) or (B, N) per-point displacement from canonical,
                       used only when grouper.motion_mode is set and cache is None
        '''
        x = features[..., :3]

        # 1. per-point local embedding (always with grad, needed by decoders)
        l = self.l_mlp(features)

        # 2. KNN neighbor cache (rebuilt by refresh(), fallback on first call)
        if self.cache is None:
            with torch.no_grad():
                neighbors, knn_idx = self.grouper(x, l, displacements)
            neighbors = neighbors.detach()
            self.cache = neighbors
            self.cache_knn_idx = knn_idx.detach()
        else:
            neighbors = self.cache

        # 3. Global latent g0 via group_encoder + transformer
        #    When g0_refresh_interval > 1, reuse cached g0 on intermediate steps
        #    (grad flows only through l/decoders; useful when encoder is the bottleneck)
        self._step_count += 1
        if self.g0_cache is None or (self._step_count % self.g0_refresh_interval == 0):
            encoded_features = self.group_encoder(neighbors)    # (B, G, H)
            encoded_features = encoded_features + self.g_pos_emb
            for block in self.blocks:
                encoded_features = block(encoded_features)
            g0 = self.g_mlp(encoded_features)                   # (B, query, query_dim)
            self.g0_cache = g0.detach()
        else:
            g0 = self.g0_cache  # detached; gradient only through l and decoders this step

        return l.squeeze(0), g0.view(self.query_dim * self.query)

    def refresh(self, features, mask=None, displacements: Optional[Tensor] = None):
        '''
        Rebuild the neighborhood cache and invalidate the g0 cache.
        displacements: (N, 3) or (N,) per-point displacement magnitudes (same masking applied)
        '''
        with torch.no_grad():
            if mask is not None:
                active_indices = torch.where(mask)[0]
                if len(active_indices) != 0:
                    features = features[active_indices]
                    if displacements is not None:
                        displacements = displacements[active_indices]
            x = features[..., :3]
            l = self.l_mlp(features)
            disp_batch = displacements.unsqueeze(0) if displacements is not None else None
            neighbors, knn_idx = self.grouper(x.unsqueeze(0), l.unsqueeze(0), disp_batch)
            neighbors = neighbors.detach()
            self.cache = neighbors
            self.cache_knn_idx = knn_idx.detach()
        self.g0_cache = None  # force g0 recompute on next forward

class LearnableTimeEmbedding(nn.Module):
    def __init__(self, K=4):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(K))  # learnable frequencies
    def forward(self, t):  # t: scalar tensor
        t = t.view(1)  # shape (1,)
        angles = 2 * torch.pi * t * self.freqs  # shape (K,)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)  # (2K,)


class ODEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, time_dim: int = 1):
        super().__init__()
        self.time_dim = time_dim
        if time_dim != 1:
            self.time_embed = LearnableTimeEmbedding(time_dim)
            time_input_ch = 2 * time_dim
        else:
            time_input_ch = 1

        self.net1 = nn.Sequential(
            nn.Linear(input_dim + time_input_ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim + time_input_ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, t: float, g: Tensor) -> Tensor:
        # g dim: (H)
        if self.time_dim==1:
            t_tensor = torch.ones(self.time_dim, device=g.device) * t
        else:
            t_tensor = self.time_embed(torch.tensor(t, device=g.device))
        h = torch.cat((g, t_tensor))
        h1 = self.net1(h)
        dzdt = self.net2(torch.cat((h1, t_tensor))) + h1
        return dzdt



class UnifiedDecoder(nn.Module):
    '''
    Single decoder for all per-point deformation outputs: v, w, s, r.

    Input: cat(z_t, l)
      z_t = l @ g_t.T  shape (N, query): local features globally modulated by
            the ODE-evolved latent g_t, already time-aware via the ODE flow.
      l   shape (N, l_dim): fine-grained per-point local embedding.
    Concatenating both gives richer features than either alone, and z_t already
    carries time information so no separate time conditioning is needed for r/s.

    Output:
      v (N,3)  linear velocity  ┐ screw axis → d_xyz via exp_se3
      w (N,3)  angular velocity ┘
      s (N,3)  delta scaling
      r (N,4)  delta rotation quaternion
    '''
    def __init__(self, z_dim: int, l_dim: int, hid_dim: int = 128, depth: int = 3):
        super().__init__()
        in_dim = z_dim + l_dim
        layers = [nn.Linear(in_dim, hid_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hid_dim, hid_dim), nn.GELU()])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hid_dim, 13)   # v(3) + w(3) + r(4) + s(3)

    def forward(self, z_t: Tensor, l: Tensor):
        h = self.mlp(torch.cat([z_t, l], dim=-1))
        out = self.head(h)                    # (N, 13)
        v = out[:, :3]
        w = out[:, 3:6]
        r = out[:, 6:10]
        s = out[:, 10:13]
        return v, w, s, r


class ODE_DisplacementModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.g_query     = config[0]["query"]
        self.g_query_dim = config[0]["l_dim"]   # always equals l_dim (not a free param)
        ode_dim = self.g_query * self.g_query_dim   # = query * l_dim
        self.encoder = KNNTransformerEncoder(**config[0])
        self.ode_func = ODEFunc(input_dim=ode_dim, hidden_dim=256)
        # decoder: cat(z_t, l) → v, w, s, r in one pass
        # z_t shape (N, query), l shape (N, l_dim)
        self.decoder = UnifiedDecoder(z_dim=config[0]["query"], l_dim=config[0]["l_dim"])
        self.steps = config[1]["steps"]
        self.odeint = odeint
        self.cache = None

    def encode(self, features: Tensor, infer: bool = False):
        """Encode scene features into local point embeddings and a global latent state."""
        if infer and self.cache is None:
            l, g0 = self.encoder(features)
            self.cache = (l, g0)
        elif infer and self.cache is not None:
            l, g0 = self.cache
        else:
            l, g0 = self.encoder(features)
        return l, g0

    def evolve_state(self, g_start: Tensor, t_start: float, t_end: float) -> Tensor:
        """Roll a latent state forward (or backward) between two timestamps."""
        t_start = float(t_start)
        t_end = float(t_end)
        if abs(t_end - t_start) < 1e-8:
            return g_start

        num_steps = max(int(self.steps), 2)
        ts = torch.linspace(t_start, t_end, steps=num_steps,
                            device=g_start.device, dtype=g_start.dtype)
        g_t = self.odeint(self.ode_func, g_start, ts, method='rk4')
        return g_t[-1]

    def decode(self, l: Tensor, g_t: Tensor):
        """Decode a latent state into deformation outputs."""
        g_t = g_t.reshape(self.g_query, self.g_query_dim).to(l.dtype)
        z_t = torch.matmul(l, g_t.t())
        return self.decoder(z_t, l)

    def forward(self, features: Tensor, t: float, infer: bool = False):
        # feature: (B, N, 14), t: (B, 1)
        # 1. encoder: latent state z0
        l, g0 = self.encode(features, infer=infer)


        # 2. NODE: g0 -> g_t
        g_t = self.evolve_state(g0, -0.01, t)

        # 3. decoder
        v, w, s, r = self.decode(l, g_t)

        return v, w, s, r
