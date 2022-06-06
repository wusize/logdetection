_base_ = '../swinb_3x_800-1400_anchor_bs2x8_coco_load.py'
num_classes = 50
# for test
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=5000,
            max_per_img=5000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            #  nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300))
)

dataset_type = 'LogDetMini'

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomRotate90', p=1),
            dict(type='ShiftScaleRotate', shift_limit=0.0625,
                 scale_limit=0.5,
                 rotate_limit=30,
                 interpolation=1,
                 p=1),
            dict(type='VerticalFlip', p=1),
            dict(type='RandomSizedBBoxSafeCrop',
                 height=800,
                 width=800,
                 interpolation=1,
                 p=1),
            dict(type='Transpose', p=1)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
            dict(
                type='RandomGamma',
                gamma_limit=(80,120),
                p=0.1),
        ],
        p=0.15),
    dict(
        type='OneOf',
        transforms=[
                dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=[0.1, 0.3],
                    contrast_limit=[0.1, 0.3],
                    p=1),
                dict(
                    type='RandomGamma',
                    gamma_limit=(80, 120),
                    p=0.1),
        ],
        p=0.1),
    dict(type='ToGray', p=0.01),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=7, p=1)
        ],
        p=0.1),
    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ImageCompression', quality_lower=99, quality_upper=100, p=1),
            dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=1),
        ],
        p=0.15),
]
ann_file = 'data/fewshotlogodetectionB/train/annotations/instances_train2017.json'
img_prefix = 'data/fewshotlogodetectionB/train/images/'
mixup = dict(type='CustomMixUp',
             prob=0.3,
             mixup=True,
             json_path=ann_file,
             img_path=img_prefix
             )

# 2048 1400 2048 1600 2048 1800 2048 2048
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(2048, 800), (2048, 1000), (2048, 1200), (2048, 1400)],
        # img_scale=[(2048, 1400), (2048, 1600), (2048, 1800), (2048, 2048)],
        img_scale=[(2048, 1024), (2048, 1280), (2048, 1536), (2048, 1792), (2048, 2048)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Add albumention
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    mixup,
    dict(
        type='Resize',
        # 2048-1400 2048-2048
        # img_scale=[(2048, 800), (2048, 1400)],
        # img_scale=[(2048, 1400), (2048, 2048)],
        img_scale=[(2048, 1024), (2048, 2048)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CustomAutoAugment', autoaug_type='v2'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

datasetA = dict(
    type=dataset_type,
    ann_file='data/fewshotlogodetectionB/train/annotations/instances_train2017.json',
    img_prefix='data/fewshotlogodetectionB/train/images',
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='ConcatDataset',
            datasets=[datasetA]
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file='data/fewshotlogodetectionB/val/annotations/instances_val2017.json',
        img_prefix='data/fewshotlogodetectionB/val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/fewshotlogodetectionB/val/annotations/instances_val2017.json',
        img_prefix='data/fewshotlogodetectionB/val/images',
        pipeline=test_pipeline)
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[8 * 12, 11 * 12])
evaluation = dict(interval=100000000000, metric='bbox', start=12 * 12 + 1)
runner = dict(type='EpochBasedRunner', max_epochs=12 * 12)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
optimizer = dict(type='AdamW', lr=0.0000125*4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))