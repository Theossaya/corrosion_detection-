# src/06_train_maskrcnn.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import cv2
import json
from tqdm import tqdm

import sys
sys.path.append(".")
from src.data.segmentation_dataset import COCOSegmentationDataset, detection_collate


def get_maskrcnn(num_classes=2, weights="DEFAULT"):
    """
    num_classes: include background (so 2 = background + corrosion)
    """
    try:
        weights_enum = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if weights == "DEFAULT" else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights_enum)
    except Exception:
        # Older API fallback
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    return model


def union_iou(pred_masks, gt_masks, threshold=0.5):
    """
    Quick-and-simple IoU: union all predicted masks vs union of all GT masks for an image.
    pred_masks: [N_pred, H, W] float (0..1) or uint8
    gt_masks:   [N_gt,   H, W] {0,1}
    """
    if pred_masks is None or len(pred_masks) == 0:
        pred_union = np.zeros_like(gt_masks[0] if len(gt_masks) else np.zeros((1,1)), dtype=np.uint8)
    else:
        if pred_masks.dtype != np.uint8:
            pred_masks = (pred_masks >= threshold).astype(np.uint8)
        pred_union = pred_masks.max(axis=0)

    if gt_masks is None or len(gt_masks) == 0:
        gt_union = np.zeros_like(pred_union, dtype=np.uint8)
    else:
        gt_union = gt_masks.max(axis=0).astype(np.uint8)

    inter = (pred_union & gt_union).sum()
    union = (pred_union | gt_union).sum()
    return (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def overlay_and_save(image, pred_masks, out_path, alpha=0.4):
    """
    Save an overlay visualization of predicted masks on the RGB image.
    """
    img = image.copy()
    if pred_masks is not None and len(pred_masks) > 0:
        union = (pred_masks.max(axis=0) >= 0.5).astype(np.uint8) * 255
        color = np.zeros_like(img)
        color[..., 1] = union  # green
        img = cv2.addWeighted(img, 1.0, color, alpha, 0)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def train_maskrcnn():
    # ---------- Paths ----------
    root = Path(__file__).resolve().parents[1]
    train_json = root / "data" / "processed" / "train_annotations.json"
    val_json   = root / "data" / "processed" / "val_annotations.json"

    ckpt_dir = root / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "maskrcnn_best.pth"

    vis_dir = root / "results" / "segmentation"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Datasets ----------
    train_ds = COCOSegmentationDataset(str(train_json), train=True)
    val_ds   = COCOSegmentationDataset(str(val_json),   train=False)

    print(f"Train images with masks: {len(train_ds)}")
    print(f"Val   images with masks: {len(val_ds)}")
    if len(train_ds) == 0:
        print("⚠ No training images with masks found. Ensure your COCO JSON contains 'segmentation'.")
        return

    # ---------- Loaders ----------
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=detection_collate)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_maskrcnn(num_classes=2, weights="DEFAULT").to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_iou = -1.0
    epochs = 8  # keep modest for CPU

    print("\n================ MASK R-CNN TRAINING ================\n")
    for epoch in range(1, epochs + 1):
        model.train()
        losses_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_epoch += loss.item()
            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()})

        lr_scheduler.step()
        avg_loss = losses_epoch / max(1, len(train_loader))
        print(f"\nEpoch {epoch} average loss: {avg_loss:.4f}")

        # ---------- Validation IoU ----------
        model.eval()
        with torch.no_grad():
            ious = []
            # also save a few visualizations
            saved = 0
            for images, targets in val_loader:
                images_cpu = [img for img in images]  # keep for viz
                images = [img.to(device) for img in images]
                outputs = model(images)

                for i in range(len(outputs)):
                    pred = outputs[i]
                    # predicted masks: [N, 1, H, W] or [N, H, W]
                    if "masks" in pred and pred["masks"].numel() > 0:
                        pm = pred["masks"].squeeze(1).cpu().numpy()
                    else:
                        pm = np.zeros((0, images_cpu[i].shape[1], images_cpu[i].shape[2]), dtype=np.float32)

                    gt = targets[i]["masks"].cpu().numpy() if targets[i]["masks"].numel() > 0 else np.zeros((0, images_cpu[i].shape[1], images_cpu[i].shape[2]), dtype=np.uint8)
                    ious.append(union_iou(pm, gt, threshold=0.5))

                    # save 3 previews per epoch
                    if saved < 3:
                        img_np = (images_cpu[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        out_path = vis_dir / f"val_epoch{epoch}_sample{saved+1}.jpg"
                        overlay_and_save(img_np, pm, out_path)
                        saved += 1

            val_iou = float(np.mean(ious)) if len(ious) else 0.0
            print(f"Validation union-IoU: {val_iou:.4f}")

            # checkpoint
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({"model_state_dict": model.state_dict(),
                            "val_union_iou": best_val_iou}, best_ckpt)
                print(f"✅ Saved best checkpoint → {best_ckpt} (IoU {best_val_iou:.4f})")

    print("\n✓ Training complete.")
    print(f"Best val union-IoU: {best_val_iou:.4f}")
    print(f"Visual previews: {vis_dir}")


if __name__ == "__main__":
    train_maskrcnn()
