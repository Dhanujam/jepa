
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_baseline import CNNBaseline


def main():
    parser = argparse.ArgumentParser(
        description="Train the supervised CNN baseline."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="cnn_baseline.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    if len(dataset.classes) < 2:
        raise ValueError("Need at least two class subfolders.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBaseline(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Classes:", dataset.classes)
    print("Images:", len(dataset))
    print("Device:", device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {total_loss / total:.4f} | "
            f"Training accuracy: {correct / total:.4f}"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
    }, args.output)

    print("Saved model to:", args.output)


if __name__ == "__main__":
    main()
