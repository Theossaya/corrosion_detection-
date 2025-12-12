from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append(".")

from src.data.pseudo_segmentation_dataset import PseudoSegmentationDataset


# -------------------------
# U-Net Model
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# -------------------------
# Loss & Metrics
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def iou_score(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    return (inter / union).mean().item()


# -------------------------
# Training
# -------------------------
def train_unet():


    root = Path(__file__).resolve().parents[1]

    train_ds = PseudoSegmentationDataset(
        coco_json=root / "data" / "processed" / "train_annotations.json",
        mask_dir=root / "data" / "pseudo_masks" / "train",
        img_size=256
    )

    val_ds = PseudoSegmentationDataset(
        coco_json=root / "data" / "processed" / "val_annotations.json",
        mask_dir=root / "data" / "pseudo_masks" / "val",
        img_size=256
    )


    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    epochs = 12

    print("\n================= U-NET TRAINING =================\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = bce(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            ious = []
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                ious.append(iou_score(preds, masks))

        val_iou = float(np.mean(ious))
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), root / "models" / "checkpoints" / "unet_best.pth")
            print("✅ Saved best U-Net model")

    print("\n✓ U-Net training complete")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    train_unet()
