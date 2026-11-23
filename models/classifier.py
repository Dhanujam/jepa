import torch
import torch.nn as nn

from models.encoder import VisionTransformerEncoder


class JEPAClassifier(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=True):
        super().__init__()

        self.encoder = VisionTransformerEncoder(
            image_size=224,
            patch_size=16,
            in_channels=1,
            embed_dim=384,
            num_heads=6,
            depth=6,
        )

        self.encoder_frozen = freeze_encoder

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        if self.encoder_frozen:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)

        features = features.mean(dim=1)

        return self.classifier(features)
