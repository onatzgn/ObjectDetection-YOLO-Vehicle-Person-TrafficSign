from ultralytics import YOLO
import pandas as pd, cv2, glob, os
from tqdm.auto import tqdm

TEMPLATE_CSV = '/content/drive/MyDrive/.../test_empty_submission_spring25.csv'
WEIGHTS_PT   = '/content/drive/MyDrive/.../yolov8m_100ep32b/weights/best.pt'
TEST_DIR     = '/content/drive/MyDrive/.../test/images'
OUT_CSV      = '/content/drive/MyDrive/.../submission_100ep32b.csv'

class_map = {0: 'pedestrian', 1: 'car', 2: 'traffic sign'}

template_df = pd.read_csv(TEMPLATE_CSV)
slots = {}
for idx, row in template_df.iterrows():
    slots.setdefault(row['img_name'], []).append(idx)

model = YOLO(WEIGHTS_PT)
image_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*')))

for img_path in tqdm(image_paths):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None or img_name not in slots: continue
    h, w = img.shape[:2]

    result = model(img_path, conf=0.001, iou=0.5, verbose=False)[0]
    for box, cid in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
        if not slots[img_name]: break
        idx = slots[img_name].pop(0)
        x1, y1, x2, y2 = box
        template_df.loc[idx, ['cls', 'x1', 'y1', 'x2', 'y2']] = [
            class_map.get(cid, f'class_{cid}'),
            max(0, min(1, x1 / w)),
            max(0, min(1, y1 / h)),
            max(0, min(1, x2 / w)),
            max(0, min(1, y2 / h))
        ]

assert list(template_df.columns) == ['ID', 'img_name', 'cls', 'x1', 'y1', 'x2', 'y2']
template_df.to_csv(OUT_CSV, index=False)
print(f' Submission hazır → {OUT_CSV}')

import os, json, shutil, random, math, pathlib, yaml
from PIL import Image
from tqdm import tqdm

ROOT = '/content/drive/MyDrive/...'
DATASET_DIR = '/content/dataset'
SRC_IMG_DIR = f'{ROOT}/train/images'
LABEL_JSON  = f'{ROOT}/train/train_data.json'
VAL_RATIO = 0.20

!rm -rf {DATASET_DIR}
os.makedirs(f'{DATASET_DIR}/images/train', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/images/val', exist_ok=True)

imgs = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.jpg','.png'))]
random.seed(42); random.shuffle(imgs)
val_set = set(imgs[: math.floor(len(imgs) * VAL_RATIO)])

for im in tqdm(imgs, desc='copy images'):
    split = 'val' if im in val_set else 'train'
    shutil.copy2(f'{SRC_IMG_DIR}/{im}', f'{DATASET_DIR}/images/{split}/{im}')

MAP = {
    'person': 'person', 'pedestrian': 'person',
    'car': 'vehicle', 'bus': 'vehicle', 'truck': 'vehicle',
    'van': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'vehicle',
    'traffic sign': 'traffic_sign', 'traffic_sign': 'traffic_sign'
}
CLASSES = ['person', 'vehicle', 'traffic_sign']
name2id = {n: i for i, n in enumerate(CLASSES)}

for split in ['train', 'val']:
    os.makedirs(f'{DATASET_DIR}/labels/{split}', exist_ok=True)

with open(LABEL_JSON) as f:
    data = json.load(f)

miss_w, miss_cls = 0, 0
for rec in tqdm(data, desc='write labels'):
    img_name = rec['name']
    split = 'val' if img_name in val_set else 'train'
    img_path = f"{DATASET_DIR}/images/{split}/{img_name}"
    if not os.path.exists(img_path): continue
    W, H = Image.open(img_path).size

    for lab in rec.get('labels', []):
        raw = lab['category'].strip().lower()
        if raw not in MAP: miss_cls += 1; continue
        cid = name2id[MAP[raw]]
        box = lab.get('bbox') or lab.get('box2d')
        x1, y1, x2, y2 = box if isinstance(box, list) else [box['x1'], box['y1'], box['x2'], box['y2']]
        xc, yc, bw, bh = (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H
        if not (0 < xc < 1 and 0 < yc < 1 and 0 < bw < 1 and 0 < bh < 1): miss_w += 1; continue
        with open(f"{DATASET_DIR}/labels/{split}/{pathlib.Path(img_name).stem}.txt", 'a') as fp:
            fp.write(f"{cid} {xc} {yc} {bw} {bh}\n")

print(f'✓ labels done | skipped (out-of-bounds: {miss_w}, unknown class: {miss_cls})')

with open('/content/data.yaml', 'w') as f:
    yaml.dump({
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASSES
    }, f, sort_keys=False)
    
from ultralytics import YOLO
import shutil

model = YOLO('yolov8m.pt')

model.train(
    data='/content/data.yaml',
    epochs=100,
    imgsz=896,
    batch=32,
    device=0,
    project='/content/runs',
    name='yolov8m_100ep32b',
    save_period=20
)

metrics = YOLO('/content/runs/yolov8m_100ep32b/weights/best.pt').val(data='/content/data.yaml')
print(f"mAP@0.5: {metrics.box.map50:.3f} | mAP@0.5:0.95: {metrics.box.map:.3f}")

DEST = f'{ROOT}/yolo_outputs/yolov8m_100ep32b'
shutil.copytree('/content/runs/yolov8m_100ep32b', DEST, dirs_exist_ok=True)
print('✓ outputs copied to Drive →', DEST)