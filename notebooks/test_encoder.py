import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.encoder import VisionTransformerEncoder


def main():

    model = VisionTransformerEncoder()

    dummy_mri = torch.randn(
        2,      # batch size
        1,      # grayscale MRI
        224,
        224
    )

    output = model(dummy_mri)

    print("Input shape :", dummy_mri.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()