import json
from pathlib import Path
import shutil

# paths
root = Path("data/processed/unified")
json_path = root / "unified_annotations.json"

with open(json_path, "r") as f:
    coco = json.load(f)

# Load earlier “top 100” file_names (paste the list you printed)
top_files = [
    "20230307_113036260_iOS_jpg.rf.70e6a24ccadc7542df2299116738a251.jpg",
    "20230307_113023177_iOS_jpg.rf.c076b7342203fee40ac1339ba69a088c.jpg",
    "20230307_112957649_iOS_jpg.rf.42e55ee8b4e5a5ef84e31d3eb782ec0c.jpg",
    # … continue with all 100 values …
]

# Build mapping image_name -> full_path
image_map = {img["file_name"]: img["path"] if "path" in img else img["file_name"]
             for img in coco["images"]}

out_dir = Path("data/segmentation_candidates")
out_dir.mkdir(parents=True, exist_ok=True)

copied = 0
for fname in top_files:
    if fname in image_map:
        src = Path(image_map[fname])
        dst = out_dir / src.name
        try:
            shutil.copy(src, dst)
            copied += 1
        except Exception as e:
            print("Failed:", src, e)

print(f"Copied {copied} images → {out_dir}")
