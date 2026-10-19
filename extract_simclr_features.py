import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from models.simclr import SimCLR
from datasets.simclr_dataset import SimCLRDataset

def main():
    root = Path("datasets/processed/ixi")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    dataset = SimCLRDataset(str(root))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = SimCLR()
    checkpoint = torch.load(
        "checkpoints/simclr_ixi_364.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    all_features = []
    all_paths = []

    with torch.no_grad():
        for start in range(0, len(dataset), 8):
            batch_paths = dataset.image_paths[start:start + 8]
            images = []
            for path in batch_paths:
                from PIL import Image
                image = Image.open(path).convert("L")
                images.append(dataset.transform(image))

            batch = torch.stack(images)
            features, _ = model(batch)
            all_features.append(features.cpu())
            all_paths.extend(str(p) for p in batch_paths)

    features = torch.cat(all_features, dim=0)
    torch.save(
        {"features": features, "image_paths": all_paths},
        output_dir / "simclr_ixi_features.pt",
    )

    with open(output_dir / "simclr_ixi_features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + [f"feature_{i}" for i in range(features.shape[1])])
        for path, vector in zip(all_paths, features.tolist()):
            writer.writerow([path] + vector)

    print("Images processed:", len(all_paths))
    print("Feature shape:", tuple(features.shape))
    print("All features finite:", bool(torch.isfinite(features).all()))
    print("Saved: outputs/simclr_ixi_features.pt")
    print("Saved: outputs/simclr_ixi_features.csv")

if __name__ == "__main__":
    main()
