file_name = "work_dirs/res_agn.bbox.json"

import json
from mmdet.datasets.api_wrappers import COCO

gt = COCO("work_dirs/instances_val2017.json")
gt = gt.loadRes(file_name)

max_cats = []
for img_id in sorted(gt.imgs):
    ann_ids = gt.getAnnIds(imgIds=img_id)
    anns = [gt.anns[_] for _ in ann_ids]
    anns = sorted(anns, key=lambda x: -x["score"])
    if anns:
        max_cat = anns[0]["category_id"]
    else:
        max_cat = -1
    max_cats.append(max_cat)

for i in range(1, len(max_cats) - 1):
    if max_cats[i - 1] == max_cats[i + 1]:
        max_cats[i] = max_cats[i - 1]

annotations = []
for i, img_id in enumerate(sorted(gt.imgs)):
    ann_ids = gt.getAnnIds(imgIds=img_id)
    anns = [gt.anns[_] for _ in ann_ids]
    anns = [_ for _ in anns if _["category_id"] == max_cats[i]]
    annotations.extend(anns)

for i in range(len(annotations)):
    annotations[i]["score"] = annotations[i]["score"] / 2 + 0.5

with open(file_name.replace(".json", ".score2.json"), "w") as f:
    json.dump(annotations, f)
