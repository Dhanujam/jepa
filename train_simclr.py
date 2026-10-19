import torch
from torch.utils.data import DataLoader

from datasets.simclr_dataset import SimCLRDataset
from models.simclr import SimCLR
from losses.nt_xent import NTXentLoss


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = SimCLRDataset("datasets/processed/ixi")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = SimCLR(projection_dim=128).to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2  # Small CPU smoke test; increase after confirming it works.
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0

        for view1, view2 in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        mean_loss = total_loss / batches if batches else float("nan")
        print(f"Epoch {epoch + 1}/{epochs} - NT-Xent loss: {mean_loss:.4f}")

    output_path = "checkpoints/simclr_ixi_364.pth"
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print("Checkpoint saved:", output_path)


if __name__ == "__main__":
    main()
