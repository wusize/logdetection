from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO


class LogDetMini(CocoDataset):
    CLASSES = (
        'FESTO', 'starbucks', 'castrol', 'hellokitty', 'SANDVIK', 'alexandermcqueen', 'balabala', 'THINKINGPUTTY', 'CCTV',
        'puma', 'piaget', 'givenchy', 'naturerepublic', 'pandora', 'chowtaiseng', 'kans', 'joyong', 'apple', 'lenovo',
        'sergiorossi', 'joeone', 'dove', 'pigeon', 'keds', 'simon', 'burberry', 'wechat', 'thermos', 'asics', 'doraemon',
        'brioni', 'rejoice', 'charles_keith', 'Jmsolution', 'musenlin', 'GOON', 'gentlemonster', 'hotwind', 'UnderArmour',
        'alibaba', 'tesla', 'gloria', 'ThomasFriends', 'fendi', 'camel', 'Thrasher', 'yuantong', 'chromehearts',
        'pepsicola', 'playboy')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        # self.CLASSES = [cat['name'] for cat in self.coco.cats.values()]
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos



