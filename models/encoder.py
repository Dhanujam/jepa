"""
JEPA
Module: Vision Transformer Encoder

This module converts a 2D MRI image into a sequence of patch
representations that can be used by the JEPA context and target encoders.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=384
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]

        x = self.projection(x)

        # [B, embed_dim, 14, 14]
        x = x.flatten(2)

        # [B, embed_dim, 196]
        x = x.transpose(1, 2)

        # [B, 196, embed_dim]
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Basic Vision Transformer encoder for JEPA.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=384,
        num_heads=6,
        depth=6,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.num_patches = self.patch_embedding.num_patches

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embedding(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        x = self.norm(x)

        return x