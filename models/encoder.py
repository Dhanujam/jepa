import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=384,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=384,
        num_heads=6,
        depth=6,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.num_patches = self.patch_embedding.num_patches
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, patch_indices=None):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        if patch_indices is None:
            positions = self.position_embedding.expand(
                batch_size, -1, -1
            )
        else:
            patch_indices = patch_indices.to(
                device=x.device, dtype=torch.long
            )
            if patch_indices.ndim != 2 or patch_indices.shape[0] != batch_size:
                raise ValueError("patch_indices must have shape [B, N]")
            if torch.any(patch_indices < 0) or torch.any(
                patch_indices >= self.num_patches
            ):
                raise ValueError("Patch index out of range")

            x = x.gather(
                1,
                patch_indices.unsqueeze(-1).expand(
                    -1, -1, x.shape[-1]
                ),
            )
            positions = self.position_embedding.expand(
                batch_size, -1, -1
            ).gather(
                1,
                patch_indices.unsqueeze(-1).expand(
                    -1, -1, self.position_embedding.shape[-1]
                ),
            )

        x = x + positions
        x = self.transformer(x)
        return self.norm(x)
