from numpy import random
import os
import numpy as np
import cv2
import random
from logdet.datasets.autoaug_utils import distort_image_with_autoaugment
from mmdet.datasets.api_wrappers import COCO


class CustomMixUp(object):
    """Mixup images & bbox
    Args:
        prob (float): the probability of carrying out mixup process.
        lambd (float): the parameter for mixup.
        mixup (bool): mixup switch.
        json_path (string): the path to dataset json file.
    """

    def __init__(self, prob=0.5, lambd=0.5, mixup=False,
                 json_path='../mmdetection/data/coco/annotations/instances_train2017.json',
                 img_path='../mmdetection/data/coco/images/'):
        self.lambd = lambd
        self.prob = prob
        self.mixup = mixup
        self.json_path = json_path
        self.img_path = img_path
        # with open(json_path, 'r') as json_file:
        #     all_labels = json.load(json_file)
        # self.all_labels = all_labels
        print(f'load coco json', flush=True)
        self.coco_json = COCO(json_path)

        self.CLASSES = [cat['name'] for cat in self.coco_json.cats.values()]
        self.cat_ids = self.coco_json.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.name2ids = {}
        for k, v in self.coco_json.imgs.items():
            self.name2ids[v['file_name']] = k

    def get_img1(self, name):
        img_id = self.name2ids[name]
        anns = self.coco_json.imgToAnns[img_id]
        labels = [np.int64(self.cat2label[annt['category_id']]) for annt in anns]
        bboxes = [[np.float32(annt['bbox'][0]),
                   np.float32(annt['bbox'][1]),
                   np.float32(annt['bbox'][0] + annt['bbox'][2] - 1),
                   np.float32(annt['bbox'][1] + annt['bbox'][3] - 1)] for annt in anns]
        img2_path = os.path.join(self.img_path, name)
        img = cv2.imread(img2_path)

        return img, labels, bboxes

    def get_img2(self):
        # random get image2 for mixup
        idx2 = np.random.choice(np.arange(len(self.coco_json.imgToAnns)))
        img_info = list(self.coco_json.imgs.values())[idx2]
        name = img_info['file_name']
        anns = list(self.coco_json.imgToAnns.values())[idx2]

        labels = [np.int64(self.cat2label[annt['category_id']]) for annt in anns]
        bboxes = [[np.float32(annt['bbox'][0]),
                   np.float32(annt['bbox'][1]),
                   np.float32(annt['bbox'][0] + annt['bbox'][2] - 1),
                   np.float32(annt['bbox'][1] + annt['bbox'][3] - 1)] for annt in anns]
        img2_path = os.path.join(self.img_path, name)
        img = cv2.imread(img2_path)

        return img, labels, bboxes

    def __call__(self, results):
        if self.mixup == True:
            if random.uniform(0, 1) > self.prob:
                name=results["img_info"]['file_name']
                img1, labels1, boxes1 = self.get_img1(name)
                img2, labels2, boxes2 = self.get_img2()
                # if labels2 != []:
                #     break

                height = max(img1.shape[0], img2.shape[0])
                width = max(img1.shape[1], img2.shape[1])

                if labels2 == []:
                    self.lambd = 0.9  # float(round(random.uniform(0.5,0.9),1))
                    # mix image
                    mixup_image = np.zeros([height, width, 3], dtype='float32')
                    mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
                    mixup_image[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - self.lambd)
                    mixup_image = mixup_image.astype('uint8')
                else:
                    # mix image
                    mixup_image = np.zeros([height, width, 3], dtype='float32')
                    mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
                    mixup_image[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - self.lambd)
                    mixup_image = mixup_image.astype('uint8')

                    # mix labels
                    results['gt_labels'] = np.hstack((np.array(labels1), np.array(labels2)))
                    results['gt_bboxes'] = np.vstack((boxes1, boxes2))

                results['img'] = mixup_image

                # if the image2 has not bboxes, the 'gt_labels' and 'gt_bboxes' need to be doubled
                # so at the end the half of loss weight can be added as 1 instead of 0.5
                # if boxes2 == []:
                #     results['gt_labels'] = np.hstack((labels1, labels1))
                #     results['gt_bboxes'] = np.vstack((list(results['gt_bboxes']), list(results['gt_bboxes'])))
                # else:

                return results
            else:
                return results

    def __repr__(self):
        return self.__class__.__name__ + \
               '(prob={}, lambd={}, mixup={}, json_path={}, img_path={})'.format(
                   self.prob, self.lambd, self.mixup, self.json_path, self.img_path)


class CustomAutoAugment(object):
    def __init__(self, autoaug_type="v1"):
        """
        Args:
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(CustomAutoAugment, self).__init__()
        self.autoaug_type = autoaug_type

    def __call__(self, results):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        gt_bbox = results['gt_bboxes']
        im = results['img']
        if len(gt_bbox) == 0:
            return results

        # gt_boxes : [x1, y1, x2, y2]
        # norm_gt_boxes: [y1, x1, y2, x2]
        height, width, _ = im.shape
        norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
        norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
        norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
        norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
        norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)

        im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
                                                          self.autoaug_type)
        gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
        gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
        gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
        gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)

        results['gt_bboxes'] = gt_bbox
        results['img'] = im
        results['img_shape'] = im.shape
        results['pad_shape'] = im.shape

        return results
