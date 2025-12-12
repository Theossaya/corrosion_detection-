import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from torch.utils.data import DataLoader
from src.data.pseudo_segmentation_dataset import PseudoSegmentationDataset

dataset = PseudoSegmentationDataset(
    coco_json="data/processed/train_annotations.json",
    mask_dir="data/pseudo_masks/train",
    img_size=256
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(loader))

print("Images:", images.shape, images.min().item(), images.max().item())
print("Masks:", masks.shape, masks.unique())
