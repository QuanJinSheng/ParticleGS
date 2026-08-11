encoder_config = (
    dict(
        in_dim=14,
        hidden_size=256,
        num_groups=4096,
        group_size=32,
        query=32,
        num_heads=4,
        mlp_ratio=4.0,
        l_dim=32
    ),
    dict(
        steps=40
    ),
)

load2gpu_on_the_fly = True
network_lr_scale = 3.0

warm_up = 3000
densify_until_iter = 20000
iterations = 60000
