
import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNBaseline(nn.Module):
    """Supervised ResNet-18 baseline for MRI image classification."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.model = resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Expected images shaped (batch, 3, height, width)"
            )
        return self.model(images)


if __name__ == "__main__":
    model = CNNBaseline(num_classes=3)
    model.eval()

    with torch.no_grad():
        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("CNN baseline test passed!")
