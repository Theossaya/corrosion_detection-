import json
from pathlib import Path

path = Path("data/processed/unified/unified_annotations.json")
with open(path, "r") as f:
    coco = json.load(f)

# Map image_id → total bbox area
img_area = {}

for ann in coco["annotations"]:
    imgid = ann["image_id"]
    x,y,w,h = ann["bbox"]
    area = w*h
    img_area[imgid] = img_area.get(imgid, 0) + area

top = sorted(img_area.items(), key=lambda x: -x[1])[:100]

# Print image filenames
img_map = {img["id"]: img for img in coco["images"]}

print("Top 100 images for segmentation:")
for imgid, area in top:
    print(img_map[imgid]["file_name"])

