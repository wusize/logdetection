_base_ = './swinb_3x_800-1400_anchor_bs2x8_albu_coco.py'

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
