_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/models/brnet.py',
    '../_base_/schedules/schedule_cos.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    rpn_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='ClassAgnosticBBoxCoder',
            num_dir_bins=12,
            with_rot=True
        ),
        objectness_loss=dict(loss_weight=10.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=5.0, beta=0.15)
    ),
    roi_head=dict(
        roi_extractor=dict(
            rep_type='ray',
            density=2,
            sa_radius=0.2,
            sa_num_sample=16,
            num_seed_points=1024),
        bbox_head=dict(
            num_classes=10,
            dir_res_loss=dict(
                type='SmoothL1Loss',
                beta=3.14 / 12,
                reduction='sum',
                loss_weight=1.0
            ),
            size_res_loss=dict(
                type='SmoothL1Loss',
                beta=0.15,
                reduction='sum',
                loss_weight=5.0
            ),
            bbox_coder=dict(
                type='ClassAgnosticBBoxCoder',
                num_dir_bins=12,
                with_rot=True
            ),
        )
    )
)

data = dict(samples_per_gpu=8)

# optimizer
lr = 0.001  # max learning rate
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