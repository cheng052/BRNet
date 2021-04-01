_base_ = [
    '../_base_/datasets/scannet-3d-18class.py',
    '../_base_/models/brnet.py',
    '../_base_/schedules/schedule_cos.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    rpn_head=dict(
        num_classes=18,
        bbox_coder=dict(
            type='ClassAgnosticBBoxCoder',
            num_dir_bins=1,
            with_rot=False
        ),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0, beta=0.15)
    ),
    roi_head=dict(
        roi_extractor=dict(
            rep_type='ray',
            density=2,
            sa_radius=0.2,
            sa_num_sample=16,
            num_seed_points=1024),
        bbox_head=dict(
            num_classes=18,
            dir_res_loss=dict(
                type='SmoothL1Loss',
                beta=3.14 / 1,
                reduction='sum',
                loss_weight=10.0
            ),
            size_res_loss=dict(
                type='SmoothL1Loss',
                beta=0.15,
                reduction='sum',
                loss_weight=10.0
            ),
            bbox_coder=dict(
                type='ClassAgnosticBBoxCoder',
                num_dir_bins=1,
                with_rot=False
            ),
        )
    )
)

data = dict(samples_per_gpu=8)

# optimizer
lr = 0.005  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

workflow = [('train', 1), ('val', 1)]
# yapf:enable

