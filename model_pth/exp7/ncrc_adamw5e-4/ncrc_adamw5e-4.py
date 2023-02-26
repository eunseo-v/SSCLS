model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=5,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(type='I3DHead', in_channels=512, num_classes=6, dropout=0.5),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file = 'data/ds_ncrc/ncrc.pkl'
left = [7, 8, 9, 11, 19, 20, 21, 22, 23, 24]
right = [3, 4, 5, 10, 13, 14, 15, 16, 17, 18]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[7, 8, 9, 11, 19, 20, 21, 22, 23, 24],
        right_kp=[3, 4, 5, 10, 13, 14, 15, 16, 17, 18]),
    dict(type='GenerateNCRCPoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GenerateNCRCPoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='GenerateNCRCPoseTarget',
        with_kp=True,
        with_limb=False,
        double=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=3),
    train=dict(
        type='RepeatDataset',
        times=30,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/ds_ncrc/ncrc.pkl',
            split='train',
            pipeline=[
                dict(type='UniformSampleFrames', clip_len=48),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(-1, 64)),
                dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
                dict(type='Resize', scale=(56, 56), keep_ratio=False),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[7, 8, 9, 11, 19, 20, 21, 22, 23, 24],
                    right_kp=[3, 4, 5, 10, 13, 14, 15, 16, 17, 18]),
                dict(
                    type='GenerateNCRCPoseTarget',
                    with_kp=True,
                    with_limb=False),
                dict(type='FormatShape', input_format='NCTHW_Heatmap'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])),
    val=dict(
        type='PoseDataset',
        ann_file='data/ds_ncrc/ncrc.pkl',
        split='test',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(64, 64), keep_ratio=False),
            dict(type='GenerateNCRCPoseTarget', with_kp=True, with_limb=False),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='data/ds_ncrc/ncrc.pkl',
        split='test',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(64, 64), keep_ratio=False),
            dict(
                type='GenerateNCRCPoseTarget',
                with_kp=True,
                with_limb=False,
                double=True),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='AdamW', lr=0.0005, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=24)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
output_config = dict(
    out='./model_pth/exp7/ncrc_adamw5e-4/test_result/results.pkl')
eval_config = dict(
    metric_out='./model_pth/exp7/ncrc_adamw5e-4/test_result',
    eval=[
        'top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix',
        't_sne_vis'
    ])
work_dir = './model_pth/exp7/ncrc_adamw5e-4'
load_from = 'model_pth/ntu120xsub_5parts_kp/best_top1_acc_epoch_23.pth'
find_unused_parameters = True
dist_params = dict(backend='nccl')
gpu_ids = range(0, 4)
