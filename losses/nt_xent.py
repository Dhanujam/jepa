
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss for SimCLR."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError("Both views must have the same shape")

        if z1.ndim != 2:
            raise ValueError("Expected embeddings shaped (batch, features)")

        batch_size = z1.shape[0]

        if batch_size < 2:
            raise ValueError("Batch size must be at least 2")

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        embeddings = torch.cat([z1, z2], dim=0)
        similarity = torch.matmul(embeddings, embeddings.T)
        similarity = similarity / self.temperature

        # Exclude self-similarity from the denominator.
        diagonal = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=embeddings.device,
        )
        similarity = similarity.masked_fill(diagonal, float("-inf"))

        # Each view's positive pair is the corresponding item in the other view.
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z1.device),
            torch.arange(batch_size, device=z1.device),
        ])

        return F.cross_entropy(similarity, targets)


if __name__ == "__main__":
    torch.manual_seed(42)

    view1 = torch.randn(4, 128, requires_grad=True)
    view2 = torch.randn(4, 128, requires_grad=True)

    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(view1, view2)
    loss.backward()

    print("NT-Xent loss:", round(loss.item(), 4))
    print("Gradients computed:", view1.grad is not None)
