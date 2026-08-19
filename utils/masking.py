"""
JEPA
Module: Patch Masking

Creates context and target patch indices for JEPA training.
"""

import torch


def create_random_masks(
    batch_size,
    num_patches,
    context_ratio=0.75,
    device="cpu"
):
    """
    Create random context and target masks.

    Parameters
    ----------
    batch_size : int
        Number of images in the batch.

    num_patches : int
        Total number of patches per image.

    context_ratio : float
        Fraction of patches visible to the context encoder.

    device : str
        Device on which tensors are created.

    Returns
    -------
    context_indices : torch.Tensor
        Indices of visible/context patches.

    target_indices : torch.Tensor
        Indices of target/masked patches.
    """

    num_context = int(num_patches * context_ratio)
    num_target = num_patches - num_context

    context_indices = []
    target_indices = []

    for _ in range(batch_size):

        permutation = torch.randperm(
            num_patches,
            device=device
        )

        context = permutation[:num_context]
        target = permutation[num_context:]

        context_indices.append(context)
        target_indices.append(target)

    context_indices = torch.stack(context_indices)
    target_indices = torch.stack(target_indices)

    return context_indices, target_indices