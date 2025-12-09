"""
04_baseline_classifier.py
Baseline Binary Classification Model for Corrosion Detection
=============================================================

• Uses transfer learning (EfficientNet, ResNet, MobileNet)
• Trains and validates on the unified dataset splits
• Evaluates and visualizes metrics + errors

Usage:
    python src/04_baseline_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

# Ensure src imports work
sys.path.append(".")

from src.data.preprocessing import CorrosionDataset, get_transforms
from src.models.baseline_classifier import BaselineClassifier


# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    "model_name": "efficientnet_b0",  # options: efficientnet_b0, resnet34, resnet50, mobilenet_v2
    "img_size": 224,
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-3,
    "patience": 5,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "paths": {
        "train_json": "data/processed/train_annotations.json",
        "val_json": "data/processed/val_annotations.json",
        "test_json": "data/processed/test_annotations.json",
        "checkpoints": "models/checkpoints",
        "results": "results",
        "figures": "results/figures"
    }
}


# =============================================================
# MODEL DEFINITION
# =============================================================


# =============================================================
# TRAINING / EVALUATION UTILITIES
# =============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


# =============================================================
# MAIN TRAINING LOOP
# =============================================================
def train_model():
    device = torch.device(CONFIG["device"])
    print(f"\n🚀 Training on device: {device}")

    # Create datasets and loaders
    train_ds = CorrosionDataset(CONFIG["paths"]["train_json"], get_transforms("train", CONFIG["img_size"]))
    val_ds = CorrosionDataset(CONFIG["paths"]["val_json"], get_transforms("val", CONFIG["img_size"]))
    test_ds = CorrosionDataset(CONFIG["paths"]["test_json"], get_transforms("test", CONFIG["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"], pin_memory=True)

    # Initialize model
    model = BaselineClassifier(CONFIG["model_name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5
)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Ensure checkpoint directory exists
    Path(CONFIG["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    best_model_path = Path(CONFIG["paths"]["checkpoints"]) / "best_baseline.pth"

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 60)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved → {best_model_path}")

    print("\n🎯 Training complete. Loading best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    evaluate_model(model, test_loader, device, history)


# =============================================================
# EVALUATION + VISUALIZATION
# =============================================================
def evaluate_model(model, loader, device, history):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n📊 Test Metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    save_results(history, all_labels, all_preds)


def save_results(history, all_labels, all_preds):
    results_dir = Path(CONFIG["paths"]["results"])
    figs_dir = Path(CONFIG["paths"]["figures"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figs_dir / "training_history.png", dpi=150, bbox_inches="tight")
    print(f"📉 Saved training curves → {figs_dir / 'training_history.png'}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Corrosion", "Corrosion"],
                yticklabels=["No Corrosion", "Corrosion"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(figs_dir / "confusion_matrix.png", dpi=150)
    print(f"📊 Saved confusion matrix → {figs_dir / 'confusion_matrix.png'}")

    # Save metrics JSON
    metrics = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved → {results_dir / 'metrics.json'}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" BASELINE CORROSION CLASSIFIER TRAINING ")
    print("=" * 70)
    train_model()
    print("\n✅ Baseline model training complete. Results available in /results/")
