"""
JEPA
Module: Exponential Moving Average

Updates the target encoder using the parameters
of the context encoder.
"""

import torch


@torch.no_grad()
def update_target_encoder(
    context_encoder,
    target_encoder,
    momentum=0.996
):
    """
    Update target encoder parameters using EMA.

    target = momentum * target + (1 - momentum) * context
    """

    for context_param, target_param in zip(
        context_encoder.parameters(),
        target_encoder.parameters()
    ):
        target_param.data.mul_(momentum)
        target_param.data.add_(
            (1.0 - momentum) * context_param.data
        )