
import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        predictor_dim=192,
        num_heads=4,
        depth=4,
        num_patches=196,
    ):
        super().__init__()

        self.num_patches = num_patches

        self.input_projection = nn.Linear(
            embed_dim, predictor_dim
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim)
        )
        nn.init.trunc_normal_(
            self.position_embedding, std=0.02
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, predictor_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth
        )

        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(
            predictor_dim, embed_dim
        )

    def forward(
        self, context_embeddings,
        context_indices, target_indices
    ):
        batch_size = context_embeddings.shape[0]

        context_indices = context_indices.long()
        target_indices = target_indices.long()

        context_positions = self.position_embedding.expand(
            batch_size, -1, -1
        ).gather(
            1,
            context_indices.unsqueeze(-1).expand(
                -1, -1, self.position_embedding.shape[-1]
            ),
        )

        target_positions = self.position_embedding.expand(
            batch_size, -1, -1
        ).gather(
            1,
            target_indices.unsqueeze(-1).expand(
                -1, -1, self.position_embedding.shape[-1]
            ),
        )

        context = (
            self.input_projection(context_embeddings)
            + context_positions
        )

        targets = (
            self.mask_token.expand(
                batch_size, target_indices.shape[1], -1
            )
            + target_positions
        )

        tokens = torch.cat([context, targets], dim=1)
        tokens = self.transformer(tokens)

        predictions = tokens[:, context.shape[1]:, :]
        predictions = self.norm(predictions)

        return self.output_projection(predictions)
