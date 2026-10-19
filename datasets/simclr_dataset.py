from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRDataset(Dataset):
    """Return two augmented views of each unlabeled MRI slice."""

    def __init__(self, image_dir="datasets/processed/ixi/slices", image_size=224):
        self.image_paths = sorted(Path(image_dir).rglob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in: {image_dir}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("L")
        return self.transform(image), self.transform(image)
