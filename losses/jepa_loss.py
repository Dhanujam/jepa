
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    JEPA loss between predicted and target patch representations.
    The target representations should come from the target encoder.
    """

    def __init__(self, normalize_targets=True):
        super().__init__()
        self.normalize_targets = normalize_targets

    def forward(self, predictions, targets):
        """
        predictions: [B, N_target, D]
        targets:     [B, N_target, D]
        """

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions={predictions.shape}, "
                f"targets={targets.shape}"
            )

        if not torch.isfinite(predictions).all():
            raise ValueError("Predictions contain NaN or Inf.")

        if not torch.isfinite(targets).all():
            raise ValueError("Targets contain NaN or Inf.")

        targets = targets.detach()

        if self.normalize_targets:
            targets = F.layer_norm(
                targets, (targets.shape[-1],)
            )

        loss = F.mse_loss(predictions, targets)

        return loss
