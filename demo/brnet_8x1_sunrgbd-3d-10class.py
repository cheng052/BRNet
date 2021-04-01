dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='IndoorPointSample', num_points=20000),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub')),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5),
            dict(type='IndoorPointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                             'dresser', 'night_stand', 'bookshelf', 'bathtub'),
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='SUNRGBDDataset',
            data_root='data/sunrgbd/',
            ann_file='data/sunrgbd/sunrgbd_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='DEPTH',
                    shift_height=True,
                    load_dim=6,
                    use_dim=[0, 1, 2]),
                dict(type='LoadAnnotations3D'),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.523599, 0.523599],
                    scale_ratio_range=[0.85, 1.15],
                    shift_height=True),
                dict(type='IndoorPointSample', num_points=20000),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                 'desk', 'dresser', 'night_stand', 'bookshelf',
                                 'bathtub')),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub'),
            filter_empty_gt=False,
            box_type_3d='Depth')),
    val=dict(
        type='SUNRGBDDataset',
        data_root='data/sunrgbd/',
        ann_file='data/sunrgbd/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(
                        type='RandomFlip3D',
                        sync_2d=False,
                        flip_ratio_bev_horizontal=0.5),
                    dict(type='IndoorPointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                     'desk', 'dresser', 'night_stand',
                                     'bookshelf', 'bathtub'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                 'night_stand', 'bookshelf', 'bathtub'),
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type='SUNRGBDDataset',
        data_root='data/sunrgbd/',
        ann_file='data/sunrgbd/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(
                        type='RandomFlip3D',
                        sync_2d=False,
                        flip_ratio_bev_horizontal=0.5),
                    dict(type='IndoorPointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                     'desk', 'dresser', 'night_stand',
                                     'bookshelf', 'bathtub'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                 'night_stand', 'bookshelf', 'bathtub'),
        test_mode=True,
        box_type_3d='Depth'))
model = dict(
    type='BRNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    rpn_head=dict(
        type='CAVoteHead',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=5.0, beta=0.15),
        num_classes=10,
        bbox_coder=dict(
            type='ClassAgnosticBBoxCoder', num_dir_bins=12, with_rot=True)),
    roi_head=dict(
        type='BRRoIHead',
        roi_extractor=dict(
            type='RepPointRoIExtractor',
            rep_type='ray',
            density=2,
            seed_feat_dim=256,
            sa_radius=0.2,
            sa_num_sample=16,
            num_seed_points=1024),
        bbox_head=dict(
            type='BRBboxHead',
            pred_layer_cfg=dict(
                in_channels=256, shared_conv_channels=(128, 128), bias=True),
            dir_res_loss=dict(
                type='SmoothL1Loss',
                beta=0.26166666666666666,
                reduction='sum',
                loss_weight=2.6),
            size_res_loss=dict(
                type='SmoothL1Loss',
                beta=0.15,
                reduction='sum',
                loss_weight=5.0),
            semantic_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
            num_classes=10,
            bbox_coder=dict(
                type='ClassAgnosticBBoxCoder', num_dir_bins=12,
                with_rot=True))),
    train_cfg=dict(
        rpn=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.3, sample_mod='seed'),
        rpn_proposal=dict(use_nms=False),
        rcnn=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.3, sample_mod='seed')),
    test_cfg=dict(
        rpn=dict(sample_mod='seed', use_nms=False),
        rcnn=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.5,
            per_class_proposal=True)))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 44
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=30,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/brnet-sunrgbd8'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
gpu_ids = range(0, 1)
