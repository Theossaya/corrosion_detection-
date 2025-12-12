import json
import os
from collections import Counter

UNIFIED_ANN = "data/processed/unified/unified_annotations.json"
IMG_DIR = "data/raw/unified"

with open(UNIFIED_ANN, "r") as f:
    data = json.load(f)

image_lookup = {img["id"]: img["file_name"] for img in data["images"]}

# Count how many annotations per image (proxy for segmentation usefulness)
count = Counter([ann["image_id"] for ann in data["annotations"]])

# Get top 200 images by annotation count
top = count.most_common(200)

print("Found", len(top), "images in unified dataset.")
print("\nReal segmentation candidates:")

for img_id, c in top:
    fname = image_lookup.get(img_id)
    if fname:
        path = os.path.join(IMG_DIR, fname)
        if os.path.exists(path):
            print(fname)
