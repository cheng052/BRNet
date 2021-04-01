model = dict(
    type='BRNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=(
            (64, 64, 128),
            (128, 128, 256),
            (128, 128, 256),
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
            loss_weight=5.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0, beta=0.15)),
    roi_head=dict(
        type='BRRoIHead',
        roi_extractor=dict(
            type='RepPointRoIExtractor',
            rep_type='ray',
            density=2,
            seed_feat_dim=256,
            sa_radius=0.2,
            sa_num_sample=16,
            num_seed_points=1024
        ),
        bbox_head=dict(
            type='BRBboxHead',
            pred_layer_cfg=dict(
                in_channels=256, shared_conv_channels=(128, 128), bias=True),
            dir_res_loss=dict(),
            size_res_loss=dict(
                type='SmoothL1Loss',
                beta=0.15,
                reduction='sum',
                loss_weight=10.0),
            semantic_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            pos_distance_thr=0.3,
            neg_distance_thr=0.3,
            sample_mod='seed'),
        rpn_proposal=dict(use_nms=False),
        rcnn=dict(
            pos_distance_thr=0.3,
            neg_distance_thr=0.3,
            sample_mod='seed')),
    test_cfg=dict(
        rpn=dict(sample_mod='seed', use_nms=False),
        rcnn=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True))
)