"""
03_loader_check.py
Dataset Loader Sanity Check + Batch Visualizer

Purpose:
    - Confirm CorrosionDataset loads correctly
    - Verify augmentation pipeline
    - Visualize sample batches from DataLoader

Usage:
    python src/03_loader_check.py
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Ensure src modules are importable
sys.path.append('.')

from src.data.preprocessing import CorrosionDataset, get_transforms

# =============================================================
# CONFIG
# =============================================================
CONFIG = {
    "train_json": "data/processed/train_annotations.json",
    "val_json": "data/processed/val_annotations.json",
    "batch_size": 8,
    "img_size": 224,
    "num_workers": 2
}

# =============================================================
# LOAD DATASETS
# =============================================================
def load_datasets():
    train_ds = CorrosionDataset(
        coco_json_path=CONFIG["train_json"],
        transform=get_transforms('train', img_size=CONFIG["img_size"]),
        classification_mode=True
    )
    val_ds = CorrosionDataset(
        coco_json_path=CONFIG["val_json"],
        transform=get_transforms('val', img_size=CONFIG["img_size"]),
        classification_mode=True
    )

    print(f"✅ Train set: {len(train_ds)} images")
    print(f"✅ Val set:   {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    return train_loader, val_loader


# =============================================================
# VISUALIZE BATCH
# =============================================================
def visualize_batch(loader, save_dir="results/figures", num_batches=1):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break

        grid_size = min(len(images), 8)
        cols = 4
        rows = int(np.ceil(grid_size / cols))

        plt.figure(figsize=(cols * 4, rows * 4))
        for idx in range(grid_size):
            img = images[idx].permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = np.clip(img, 0, 1)

            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img)
            label = "Corrosion" if labels[idx] == 1 else "No Corrosion"
            plt.title(label, fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        out_path = Path(save_dir) / f"loader_batch_{i+1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"🖼️ Saved batch visualization → {out_path}")
        plt.close()


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" DATASET LOADER SANITY CHECK")
    print("=" * 60)

    train_loader, val_loader = load_datasets()
    visualize_batch(train_loader, num_batches=2)

    print("\n✅ Sanity check complete — Dataloader works properly.")
    print("Next: proceed to baseline model training.")
    print("=" * 60)
