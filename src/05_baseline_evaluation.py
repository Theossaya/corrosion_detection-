"""
Evaluate Baseline Corrosion Classifier
-------------------------------------
Loads best checkpoint and evaluates on test set.
Generates confusion matrix, classification report, and accuracy summary.

Usage:
    python src/05_baseline_evaluation.py
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import numpy as np
import sys

sys.path.append(".")
from src.data.preprocessing import CorrosionDataset, get_transforms
from src.models.baseline_classifier import BaselineClassifier


def evaluate_baseline():
    print("=" * 70)
    print(" " * 20 + "BASELINE MODEL EVALUATION")
    print("=" * 70)

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    root = Path(__file__).resolve().parents[1]
    test_json = root / "data" / "processed" / "test_annotations.json"
    checkpoint_path = root / "models" / "checkpoints" / "best_baseline.pth"
    results_dir = root / "results" / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Dataset & loader
    # -------------------------------------------------
    print(f"\n📂 Loading test dataset: {test_json}")
    test_dataset = CorrosionDataset(
        coco_json_path=str(test_json),
        transform=get_transforms("test", img_size=224),
        classification_mode=True
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
              
    model = BaselineClassifier(model_name="efficientnet_b0", num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"✅ Loaded weights from {checkpoint_path}")

    model.to(device)
    model.eval()


    # -------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model.model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=["No Corrosion", "Corrosion"],
        digits=4
    )
    print("\n" + report)

    cm = confusion_matrix(all_labels, all_preds)

    # -------------------------------------------------
    # Save artifacts
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Corrosion", "Corrosion"],
                yticklabels=["No Corrosion", "Corrosion"])
    plt.title("Confusion Matrix - Baseline Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": acc, "confusion_matrix": cm.tolist()}, f, indent=2)

    print(f"\n📊 Results saved in: {results_dir}")
    print("=" * 70)
    print("✓ Evaluation complete.\n")


if __name__ == "__main__":
    evaluate_baseline()
