
import copy
import torch

from models.encoder import VisionTransformerEncoder
from models.predictor import JEPAPredictor
from utils.masking import (
    create_random_masks,
    create_tissue_aware_masks,
)
from utils.ema import update_target_encoder
from losses.jepa_loss import JEPALoss


class JEPATrainer:
    def __init__(
        self,
        device=None,
        learning_rate=1e-4,
        momentum=0.996,
        masking_strategy="random",
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if masking_strategy not in ("random", "tissue_aware"):
            raise ValueError(
                "masking_strategy must be 'random' or 'tissue_aware'"
            )

        self.masking_strategy = masking_strategy

        self.context_encoder = VisionTransformerEncoder().to(self.device)

        self.target_encoder = copy.deepcopy(
            self.context_encoder
        ).to(self.device)

        self.predictor = JEPAPredictor().to(self.device)

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

        self.loss_fn = JEPALoss()

        self.optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters()),
            lr=learning_rate,
        )

        self.momentum = momentum
        self.step = 0

    def train_step(self, images):
        images = images.to(self.device)

        self.context_encoder.train()
        self.predictor.train()
        self.target_encoder.eval()

        # Choose the masking strategy.
        if self.masking_strategy == "random":
            context_indices, target_indices = create_random_masks(
                batch_size=images.shape[0],
                num_patches=self.context_encoder.num_patches,
                context_ratio=0.75,
                device=self.device,
            )
        else:
            context_indices, target_indices = create_tissue_aware_masks(
                images=images,
                context_ratio=0.75,
                patch_size=16,
                device=self.device,
            )

        # Encode only the visible context patches.
        context_embeddings = self.context_encoder(
            images,
            patch_indices=context_indices,
        )

        # Generate target representations without gradients.
        with torch.no_grad():
            target_embeddings = self.target_encoder(images)

            target_embeddings = target_embeddings.gather(
                1,
                target_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    target_embeddings.shape[-1],
                ),
            )

        # Predict representations for the masked target patches.
        predictions = self.predictor(
            context_embeddings,
            context_indices,
            target_indices,
        )

        loss = self.loss_fn(predictions, target_embeddings)

        # Update the context encoder and predictor.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters()),
            max_norm=1.0,
        )

        self.optimizer.step()

        # Update the target encoder using EMA.
        update_target_encoder(
            self.context_encoder,
            self.target_encoder,
            momentum=self.momentum,
        )

        self.step += 1

        return loss.item()

    def save_checkpoint(self, path):
        torch.save(
            {
                "step": self.step,
                "masking_strategy": self.masking_strategy,
                "context_encoder": self.context_encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
