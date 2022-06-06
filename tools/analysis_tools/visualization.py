import numpy as np
from mmdet.datasets.api_wrappers import COCO
import os
import cv2
gt_coco = COCO('data/logdet_data/instances_val2017.json')
res_coco = gt_coco.loadRes('work_dirs/debug.bbox.json')

img_prefix = 'data/logdet_data/images'
save_dir = 'data/logdet_data/vis'
gt_save_dir = 'data/logdet_data/vis_gt'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(gt_save_dir, exist_ok=True)
thr = 0.1
for img_id, anns in res_coco.imgToAnns.items():
    image_name = res_coco.imgs[img_id]['file_name']
    save_path = os.path.join(save_dir, image_name)
    gt_save_path = os.path.join(gt_save_dir, image_name)
    image = cv2.imread(os.path.join(img_prefix, image_name),
                       cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    gt_image = cv2.imread(os.path.join(img_prefix, image_name),
                          cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    for ann in anns:
        if ann['score'] > thr:
            x, y, w, h = ann['bbox']
            image = cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)),
                                  color=(0, 0, 255), thickness=2)
            cv2.putText(image, str(ann['category_id']), (int(x), int(y + h/2)),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)

    gts = gt_coco.imgToAnns[img_id]
    for gt in gts:
        x, y, w, h = gt['bbox']
        gt_image = cv2.rectangle(gt_image, (int(x), int(y)), (int(x + w), int(y + h)),
                                 color=(0, 0, 255), thickness=2)
        cv2.putText(gt_image, str(gt['category_id']), (int(x), int(y + h / 2)),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imwrite(save_path, np.concatenate([image, gt_image], axis=1))
    # cv2.imwrite(gt_save_path, gt_image)
