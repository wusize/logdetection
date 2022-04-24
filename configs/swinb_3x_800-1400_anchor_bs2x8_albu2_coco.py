_base_ = './swinb_3x_800-1400_anchor_bs2x8_coco_load.py'


albu_train_transforms = [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
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
                        p=1.0)
                ],
                p=0.1),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                # 提点
                type='OneOf',
                transforms=[
                    dict(type='MotionBlur', p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomFog', fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1.0),
                    dict(type='RandomRain', brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1.0)
                ],
                p=0.1),
            dict(type='GaussNoise', p=0.1)
]


dataset_type = 'LogDetMini'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2048, 800), (2048, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
    # Add albumention
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
    ann_file='data/fewshotlogodetection/train/annotations/instances_train2017.json',
    img_prefix='data/fewshotlogodetection/train/images',
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            datasets=[datasetA]
        )
    ),
)
optimizer = dict(type='AdamW', lr=0.0000125*4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))