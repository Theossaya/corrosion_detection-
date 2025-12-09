"""
02_dataset_summary.py
Quick summary and integrity check for corrosion dataset
"""

import json
from pathlib import Path
from collections import Counter

# =============================================================
# PATHS
# =============================================================
try:
    ROOT = Path(__file__).resolve().parents[1]  # project root
except NameError:
    # fallback for notebooks
    from os import getcwd
    ROOT = Path(getcwd())

DATA_DIR = ROOT / "data" / "processed"
UNIFIED_JSON = DATA_DIR / "unified" / "unified_annotations.json"
SPLIT_FILES = {
    "train": DATA_DIR / "train_annotations.json",
    "val": DATA_DIR / "val_annotations.json",
    "test": DATA_DIR / "test_annotations.json",
}

# =============================================================
# UTILS
# =============================================================
def load_json(path):
    if not path.exists():
        print(f"❌ Missing file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def summarize_dataset(coco_data, name):
    if not coco_data:
        return
    imgs = coco_data.get("images", [])
    anns = coco_data.get("annotations", [])
    cats = coco_data.get("categories", [])
    cat_names = {c["id"]: c["name"] for c in cats}

    print(f"\n📁 {name.upper()} SUMMARY")
    print("-" * 60)
    print(f"🖼️  Images: {len(imgs)}")
    print(f"🔖  Annotations: {len(anns)}")
    print(f"🏷️  Categories: {len(cats)} → {list(cat_names.values())}")

    # Count per category
    counts = Counter([a["category_id"] for a in anns])
    for cid, count in counts.items():
        print(f"   • {cat_names.get(cid, 'unknown')}: {count}")

    # Check for missing image paths
    missing = [i for i in imgs if not Path(i.get("path", i["file_name"])).exists()]
    if missing:
        print(f"⚠️  Missing {len(missing)} image files:")
        for m in missing[:5]:
            print("   -", m["file_name"])
    else:
        print("✅ All image files exist")

# =============================================================
# RUN
# =============================================================
print(f"📂 Project root: {ROOT}")
print(f"📄 Unified file: {UNIFIED_JSON}")

unified = load_json(UNIFIED_JSON)
summarize_dataset(unified, "unified")

for name, path in SPLIT_FILES.items():
    data = load_json(path)
    summarize_dataset(data, name)

print("\n✅ Dataset summary complete.")
