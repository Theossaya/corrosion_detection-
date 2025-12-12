import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_masks(coco_json, split):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "pseudo_masks" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    print(f"Generating pseudo masks for {split}...")

    for img_id, img in tqdm(images.items()):
        h, w = img["height"], img["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns_by_img.get(img_id, []):
            x, y, bw, bh = map(int, ann["bbox"])
            mask[y:y+bh, x:x+bw] = 1

        if mask.sum() == 0:
            continue

        name = Path(img["file_name"]).stem
        cv2.imwrite(str(out_dir / f"{name}_mask.png"), mask * 255)

    print(f"✓ Masks saved → {out_dir}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_masks(root / "data/processed/train_annotations.json", "train")
    generate_masks(root / "data/processed/val_annotations.json", "val")
