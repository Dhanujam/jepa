
import torch
import torch.nn as nn


class BrainTumorClassifier(nn.Module):
    """
    Classification head for JEPA-learned MRI representations.

    Input:  (batch_size, embedding_dim)
    Output: (batch_size, num_classes) logits
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                "Expected embeddings with shape (batch_size, embedding_dim)"
            )

        return self.classifier(embeddings)


if __name__ == "__main__":
    model = BrainTumorClassifier(embedding_dim=384, num_classes=3)
    sample_embeddings = torch.randn(4, 384)
    logits = model(sample_embeddings)

    print("Input shape:", sample_embeddings.shape)
    print("Output shape:", logits.shape)
    print("Predicted class indices:", logits.argmax(dim=1))
