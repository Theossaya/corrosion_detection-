"""
01_eda_and_split.py
Exploratory Data Analysis + Train/Val/Test Split
for the unified corrosion dataset
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split

# =============================================================
# PATH SETUP (FIXED)
# =============================================================
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(os.getcwd()).resolve()

# Detect top-level corrosion_detection folder automatically
project_root = ROOT
while project_root.name.lower() != "corrosion_detection" and project_root.parent != project_root:
    project_root = project_root.parent

DATA_DIR = project_root / "data"
UNIFIED_JSON = DATA_DIR / "processed" / "unified" / "unified_annotations.json"
RESULTS_DIR = project_root / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Detected project root: {project_root}")
print(f"📄 Loading unified dataset from: {UNIFIED_JSON}")

if not UNIFIED_JSON.exists():
    raise FileNotFoundError(f"Unified annotations not found at: {UNIFIED_JSON}")

# =============================================================
# LOAD COCO JSON
# =============================================================
with open(UNIFIED_JSON, "r") as f:
    coco_data = json.load(f)

images = coco_data["images"]
annotations = coco_data["annotations"]
categories = coco_data["categories"]
print(f"✅ Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories.")

# =============================================================
# EDA: CATEGORY COUNTS
# =============================================================
cat_counts = defaultdict(int)
for ann in annotations:
    cat_counts[ann["category_id"]] += 1

cat_names = {c["id"]: c["name"] for c in categories}

plt.figure(figsize=(6, 4))
plt.bar([cat_names[k] for k in cat_counts.keys()], cat_counts.values(), color="steelblue")
plt.title("Annotation Counts per Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "data_distributions.png", dpi=300)
plt.close()

print(f"📊 Saved category distribution plot → {RESULTS_DIR / 'data_distributions.png'}")

# =============================================================
# VISUALIZE RANDOM SAMPLES
# =============================================================
def show_random_samples(coco_json_path, num_samples=6):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {c['id']: c['name'] for c in coco_data['categories']}

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    samples = random.sample(images, min(num_samples, len(images)))

    plt.figure(figsize=(16, 10))
    for idx, img_info in enumerate(samples):
        img_path = Path(img_info.get("path", "")) or Path(img_info["file_name"])
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anns = img_to_anns[img_info["id"]]

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cat_name = categories.get(ann["category_id"], "corrosion")
            cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        plt.subplot(2, (num_samples + 1)//2, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{Path(img_info['file_name']).stem} ({len(anns)} boxes)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_annotations.png", dpi=300)
    plt.show()

show_random_samples(UNIFIED_JSON, num_samples=6)
print(f"🖼️ Saved sample visualization → {RESULTS_DIR / 'sample_annotations.png'}")

# =============================================================
# TRAIN/VAL/TEST SPLIT
# =============================================================
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

for split_name, split_imgs in splits.items():
    img_ids = {img["id"] for img in split_imgs}
    split_anns = [a for a in annotations if a["image_id"] in img_ids]
    out_data = {
        "images": split_imgs,
        "annotations": split_anns,
        "categories": categories
    }
    out_path = DATA_DIR / "processed" / f"{split_name}_annotations.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"💾 Saved {split_name} set → {out_path} ({len(split_imgs)} imgs, {len(split_anns)} anns)")

print("✅ EDA and dataset split complete.")
