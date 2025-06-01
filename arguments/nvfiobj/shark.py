grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,

    bound=1.6,
)

encoder_config = dict(
    in_channels=35,
    hidden_size=128,
    num_groups=4096,
    group_size=32,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0
)

grid_lr_scale = 50.0
network_lr_scale = 5.0

warm_up = 3000
densify_until_iter = 20000
