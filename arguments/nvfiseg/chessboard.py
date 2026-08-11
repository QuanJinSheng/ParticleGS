encoder_config = (
    dict(
        in_dim=14,
        hidden_size=256,
        num_groups=2048,
        group_size=32,
        query=4,
        num_heads=4,
        mlp_ratio=4.0,
        l_dim=32
    ),
    dict(
        steps=20
    ),
)

network_lr_scale = 3.0

warm_up = 3000
iterations = 60000


