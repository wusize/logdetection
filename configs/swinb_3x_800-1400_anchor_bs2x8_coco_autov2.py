_base_ = './swinb_3x_800-1400_anchor_bs2x8_coco_load.py'

# for test
model = dict(
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
            max_per_img=300)))

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
