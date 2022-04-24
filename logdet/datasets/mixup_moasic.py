import json
import numpy as np
import cv2
import random
import math


class CustomMosaic(object):
    def __init__(self, prob=0.5, img_size=1200,
                 json_path='data/coco/annotations/instrance_train2017.json',
                 img_path='data/coco/train2017/'):
        self.prob = prob
        self.img_size = img_size
        self.json_path = json_path
        self.img_path = img_path
        with open(json_path, 'r') as json_file:
            all_labels = json.load(json_file)
        self.all_labels = all_labels

    def get_img2(self,index):
        # random get image2 for mixup
        idx2 = index
        img2_fn = self.all_labels['images'][idx2]['file_name']
        img2_id = self.all_labels['images'][idx2]['id']
        img2_path = self.img_path + img2_fn
        img2 = cv2.imread(img2_path)
        h,w,c=img2.shape
        boxes2 = []
        for annt in self.all_labels['annotations']:
            if annt['image_id'] == img2_id:
                boxes2.append([np.int64(annt['category_id']),
                               np.float32((annt['bbox'][0] + annt['bbox'][2] / 2.0) / w),
                               np.float32((annt['bbox'][1] + annt['bbox'][3] / 2.0) / h),
                               np.float32(annt['bbox'][2] / w),
                               np.float32(annt['bbox'][3] / h)])
        return img2,np.array(boxes2),h,w

    def random_affine(self, img, targets=None, degrees=10, translate=.1, scale=.1, shear=10, border=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        if targets is None:  # targets = [cls, xyxy]
            targets = np.array([]).astype(np.int64)
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        imgnn=img

        return imgnn, targets

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            s=self.img_size
            xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
            indice = []
            labels4=[]
            name=results['filename']
            idx1 = 0
            for i in range(len(self.all_labels['images'])):
                if self.all_labels['images'][i]['file_name'] == name:
                    idx1 = self.all_labels['images'][i]['id']
            indice.append(idx1)
            for i in range(3):
                idx2 = np.random.choice(np.arange(len(self.all_labels['images'])))
                indice.append(idx2)
            for i in range(4):
                img, label, h, w=self.get_img2(indice[i])
                if i == 0:  # top left
                    img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                             0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b
                d=label.copy()
                if label.size > 0:  # Normalized xywh to pixel xyxy format
                    d[:, 1] = w * (label[:, 1] - label[:, 3] / 2) + padw
                    d[:, 2] = h * (label[:, 2] - label[:, 4] / 2) + padh
                    d[:, 3] = w * (label[:, 1] + label[:, 3] / 2) + padw
                    d[:, 4] = h * (label[:, 2] + label[:, 4] / 2) + padh
                labels4.append(d)
            if len(labels4):
                labels4 = np.concatenate(labels4, 0)
                # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
                np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
            else:
                labels4 = None
            img4, labels4 = self.random_affine(img4, labels4,
                                               degrees=1.98 * 2,
                                          translate=0.05 * 2,
                                          scale=0.05 * 2,
                                          shear=0.641 * 2,
                                          border=-s // 2)
            labelnn=labels4[:,0]
            boxesnn=labels4[:,1:]
            results['gt_labels']=labelnn.astype(np.int64)
            results['gt_bboxes']=boxesnn.astype(np.float32)
            results['img'] = img4
            return  results
        else:
            return results

    def __repr__(self):
        return self.__class__.__name__ + '(prob={},  img_size={}, json_path={}, img_path={})'.\
            format(self.prob,
               self.img_size,
               self.json_path,
               self.img_path)








