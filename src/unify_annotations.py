"""
Unified Annotation Converter for Organized Datasets
---------------------------------------------------
Automatically detects annotation types (YOLO / COCO / VOC / PNG mask)
and merges all into a single unified COCO JSON dataset.

Usage:
    python unify_annotations.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import xml.etree.ElementTree as ET


class UnifiedAnnotationBuilder:
    def __init__(self, organized_dir="data/organized", output_dir="data/processed/unified"):
        self.organized_dir = Path(organized_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.coco = {
            "info": {
                "description": "Unified Corrosion Detection Dataset",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": []
        }

        self.image_id = 1
        self.ann_id = 1
        self.category_map = {}

    # ---------------------------------------------------
    # CATEGORY HANDLER
    # ---------------------------------------------------
    def add_category(self, name):
        """Ensure each class has a unique ID in COCO categories."""
        name = name.lower().strip()
        if name not in self.category_map:
            cid = len(self.category_map) + 1
            self.category_map[name] = cid
            self.coco["categories"].append({
                "id": cid,
                "name": name,
                "supercategory": "defect"
            })
        return self.category_map[name]

    # ---------------------------------------------------
    # YOLO HANDLER
    # ---------------------------------------------------
    def convert_yolo(self, ann_dir, img_dir):
        print("  🟢 Converting YOLO format...")
        label_files = list(Path(ann_dir).glob("*.txt"))

        for label_file in tqdm(label_files):
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = Path(img_dir) / f"{label_file.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if not img_path:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            self.coco["images"].append({
                "id": self.image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
                "path": str(img_path)
            })

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, x, y, bw, bh = parts
                    x, y, bw, bh = map(float, [x, y, bw, bh])
                    x, y, bw, bh = x * w - bw * w / 2, y * h - bh * h / 2, bw * w, bh * h

                    cat_id = self.add_category("corrosion")
                    self.coco["annotations"].append({
                        "id": self.ann_id,
                        "image_id": self.image_id,
                        "category_id": cat_id,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0
                    })
                    self.ann_id += 1
            self.image_id += 1

    # ---------------------------------------------------
    # COCO HANDLER
    # ---------------------------------------------------
    def convert_coco(self, ann_file, img_dir):
        print(f"  🟣 Converting COCO JSON: {ann_file.name}")
        with open(ann_file, "r") as f:
            data = json.load(f)

        id_map = {}
        for img in data.get("images", []):
            img_path = Path(img_dir) / img["file_name"]
            if not img_path.exists():
                continue
            self.coco["images"].append({
                "id": self.image_id,
                "file_name": img["file_name"],
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "path": str(img_path)
            })
            id_map[img["id"]] = self.image_id
            self.image_id += 1

        for cat in data.get("categories", []):
            self.add_category(cat["name"])

        for ann in data.get("annotations", []):
            if ann["image_id"] not in id_map:
                continue
            new_ann = {
                "id": self.ann_id,
                "image_id": id_map[ann["image_id"]],
                "category_id": ann["category_id"],
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0)
            }
            if "segmentation" in ann:
                new_ann["segmentation"] = ann["segmentation"]
            self.coco["annotations"].append(new_ann)
            self.ann_id += 1

    # ---------------------------------------------------
    # VOC HANDLER
    # ---------------------------------------------------
    def convert_voc(self, xml_files, img_dir):
        print("  🔵 Converting Pascal VOC XML...")
        for xml_file in tqdm(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename").text
            size = root.find("size")
            w, h = int(size.find("width").text), int(size.find("height").text)

            img_path = Path(img_dir) / filename
            if not img_path.exists():
                continue

            self.coco["images"].append({
                "id": self.image_id,
                "file_name": filename,
                "width": w,
                "height": h,
                "path": str(img_path)
            })

            for obj in root.findall("object"):
                name = obj.find("name").text
                cat_id = self.add_category(name)
                box = obj.find("bndbox")
                xmin, ymin = float(box.find("xmin").text), float(box.find("ymin").text)
                xmax, ymax = float(box.find("xmax").text), float(box.find("ymax").text)
                bw, bh = xmax - xmin, ymax - ymin

                self.coco["annotations"].append({
                    "id": self.ann_id,
                    "image_id": self.image_id,
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })
                self.ann_id += 1
            self.image_id += 1

    # ---------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------
    def unify_all(self):
        datasets = [p for p in self.organized_dir.iterdir() if p.is_dir()]

        for ds in datasets:
            ann_dir = ds / "annotations"
            img_dir = ds / "images"

            if not ann_dir.exists() or not img_dir.exists():
                continue

            # Look for typical annotation patterns
            yolo = list(ann_dir.glob("*.txt"))
            coco = list(ann_dir.glob("*annotations*.json"))  # detect Roboflow-style
            voc = list(ann_dir.glob("*.xml"))
            masks = list(ann_dir.glob("*.png"))

            print(f"\n📁 Processing {ds.name} ...")

            try:
                if yolo:
                    print(f"  🟢 Detected YOLO format ({len(yolo)} labels)")
                    self.convert_yolo(ann_dir, img_dir)

                elif coco:
                    print(f"  🟣 Detected COCO format ({len(coco)} files)")
                    for cfile in coco:
                        self.convert_coco(cfile, img_dir)

                elif voc:
                    print(f"  🔵 Detected Pascal VOC format ({len(voc)} xml files)")
                    self.convert_voc(voc, img_dir)

                elif masks:
                    print(f"  🟠 Detected mask images (segmentation) — skipping for now")

                else:
                    print("  ⚠️ No recognizable annotation format found")
            except Exception as e:
                print(f"  ✗ Error processing {ds.name}: {e}")

        # Save unified JSON
        output = self.output_dir / "unified_annotations.json"
        with open(output, "w") as f:
            json.dump(self.coco, f, indent=2)

        print(f"\n✅ Unified dataset written to {output}")
        print(f"Total images: {len(self.coco['images'])}")
        print(f"Total annotations: {len(self.coco['annotations'])}")
        print(f"Categories: {list(self.category_map.keys())}")


# ---------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------
if __name__ == "__main__":
    builder = UnifiedAnnotationBuilder()
    builder.unify_all()
