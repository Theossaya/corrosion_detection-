# corrosion_detection-
Perfect — this will serve as your **Day 1 Project Log** for *Corrosion Detection AI*.
Here’s a clear, factual summary structured like a progress report you can keep in your repo or README.

---

#  Dataset Preparation & Baseline Classification Report**

**Project:** Corrosion Detection AI
**Date:** [Insert current date]
**Engineer:** Eric Favour
**Environment:** Windows 10 (venv, CPU mode)
**Frameworks:** PyTorch 2.8 + TorchVision 0.15 + Albumentations 1.3

---

## 1️ Dataset Preparation

###  **Unified Dataset**

* **Source:** Multiple corrosion annotation datasets merged into a single COCO-style file
  → `data/processed/unified/unified_annotations.json`
* **Images:** 561
* **Annotations:** 3056
* **Classes:** 1 (“corrosion”)

###  **EDA & Visualization**

Executed `src/01_eda_and_split.py`

**Generated plots:**

* `results/figures/data_distributions.png` — annotation count histogram
* `results/figures/sample_annotations.png` — random sample visualization

**Findings:**

* Images vary in background and texture.
* Class imbalance negligible since single class.
* Several fine-grain corrosion spots visible, good data quality.

###  **Dataset Splitting**

Random stratified split based on image IDs:

| Split      | Images | Annotations |
| ---------- | ------ | ----------- |
| Train      | 448    | 2309        |
| Validation | 56     | 338         |
| Test       | 57     | 409         |

Outputs saved under:

* `data/processed/train_annotations.json`
* `data/processed/val_annotations.json`
* `data/processed/test_annotations.json`

All files validated for missing or broken image links → **No missing files.**

---

## 2️ Baseline Classifier

**Script:** `src/04_baseline_classifier.py`
**Architecture:** EfficientNet-B0 (pretrained on ImageNet)
**Task:** Binary classification (Corrosion vs No Corrosion)
**Training device:** CPU
**Image size:** 224 × 224
**Batch size:** 16
**Epochs:** 30
**Optimizer:** Adam (lr = 1e-3)
**Scheduler:** ReduceLROnPlateau (factor = 0.5, patience = 2)
**Loss:** CrossEntropyLoss
**Early Stopping:** patience = 5
**Augmentations:** Random crop, flip, brightness/contrast, GaussNoise

---

## 3️ Training Results

| Metric     | Best Value (Validation) | Final Value (Test) |
| ---------- | ----------------------- | ------------------ |
| Accuracy   | 92.9 %                  | 91.2 %             |
| Precision  | — (Val)                 | 0.9107             |
| Recall     | — (Val)                 | 1.0000             |
| F1 Score   | — (Val)                 | 0.9533             |
| Loss (Val) | 0.1605                  | 0.2202             |

**Confusion Matrix (Test):**

|                       | Pred No Corrosion | Pred Corrosion |
| --------------------- | ----------------- | -------------- |
| **True No Corrosion** | 1                 | 5              |
| **True Corrosion**    | 0                 | 51             |

**Observation:** Model is very sensitive to corrosion presence — good for detection, but slightly over-flags corrosion (recall = 1.0, precision = 0.91).

---

## 4️ Artifacts Generated

| Type              | File Path                                      |
| ----------------- | ---------------------------------------------- |
| Training Curve    | `results/figures/training_history.png`         |
| Confusion Matrix  | `results/figures/confusion_matrix.png`         |
| Test Metrics      | `results/metrics.json`                         |
| Model Checkpoint  | `models/checkpoints/best_baseline.pth`         |
| Evaluation Report | `results/evaluation/classification_report.txt` |

---

## 5️ Insights & Next Steps

**Strengths:**

* High recall → good at catching corrosion spots.
* Consistent train/val/test performance → no overfitting.
* Pipeline structure validated (end-to-end).

**Weaknesses:**

* Over-sensitivity to corrosion (class imbalance in support).
* No segmentation yet → localization uncertain.

**Next Milestone :**

1. Generate pixel-level masks from annotations.
2. Implement U-Net baseline segmentation model.
3. Evaluate IoU and visual overlays.



