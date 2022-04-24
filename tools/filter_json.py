import json
from pycocotools.coco import COCO
with open(r'F:\work_dirs\models\JSON_coco_auto.bbox.json', 'r') as f:
    data = json.load(f)
new_data = []
# for d in data:
#     if d['score'] >= 0.5:
#         new_data.append(d)
for d in data:
    d['score'] = (d['score'] + 1.0) / 2.0
    new_data.append(d)

with open(r'models\JSON_coco_auto.bbox.json', 'w') as f:
    json.dump(new_data, f)
