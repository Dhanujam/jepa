
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class IXIMRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MRI dataset directory not found: {self.data_dir}"
            )

        # Collect PNGs only from folders named "slices".
        self.image_paths = sorted(
            path
            for folder in self.data_dir.rglob("slices")
            if folder.is_dir()
            for path in folder.glob("*.png")
        )

        if not self.image_paths:
            raise ValueError(
                f"No PNG slices found inside {self.data_dir}"
            )

        print(f"Found {len(self.image_paths)} MRI PNG slices.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize((224, 224))
            array = np.array(image, dtype=np.float32) / 255.0

        # Shape: [1, 224, 224]
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor
