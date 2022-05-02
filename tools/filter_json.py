import json
filename = 'JSON_coco_albu_auto_mixup_advance_w2_6.bbox.json'
with open(f'F:\work_dirs\models\{filename}', 'r') as f:
    data = json.load(f)
new_data = []
# for d in data:
#     if d['score'] >= 0.5:
#         new_data.append(d)
for d in data:
    d['score'] = (d['score'] + 1.0) / 2.0
    new_data.append(d)

with open(f'checkpoints\{filename}', 'w') as f:
    json.dump(new_data, f)
