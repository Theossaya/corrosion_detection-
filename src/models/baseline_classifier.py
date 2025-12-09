import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import json
import torchvision.models as models


class BaselineClassifier(nn.Module):
    """Transfer learning baseline model"""
    def __init__(self, model_name="efficientnet_b0", num_classes=2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._load_pretrained_model()

    def _load_pretrained_model(self):
        if self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)

        elif self.model_name == "resnet34":
            model = models.resnet34(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)

        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)

        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return model

    def forward(self, x):
        return self.model(x)
