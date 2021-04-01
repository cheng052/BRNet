# optimizer
# This schedule is mainly used by models in indoor dataset,
# e.g. BRNet on SUNGRBD and ScanNet

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0
)
# runtime settings
total_epochs = 44