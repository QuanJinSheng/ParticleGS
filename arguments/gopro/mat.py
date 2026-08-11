
encoder_config = (
    dict(
        in_dim=14,
        hidden_size=256,
        num_groups=8000,
        group_size=32,
        query=16,
        num_heads=4,
        mlp_ratio=6.0,
        l_dim=32
    ),
    dict(
        steps=80
    ),
)

load2gpu_on_the_fly = True
network_lr_scale = 1.0

warm_up = 3000
densify_until_iter = 15000
iterations = 60000
