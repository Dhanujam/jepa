import torch
import torch.nn as nn
from torchvision import models


class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        # ResNet18 encoder
        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return features, projections