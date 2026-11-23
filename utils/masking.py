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

def create_tissue_aware_masks(
    images,
    context_ratio=0.75,
    patch_size=16,
    device=None,
):
    """
    Create context and target indices biased toward patches
    containing non-background image content.

    This uses mean absolute intensity as a rough proxy, not
    validated brain-tissue segmentation.

    Parameters
    ----------
    images : torch.Tensor
        MRI batch with shape [B, 1, H, W].

    context_ratio : float
        Fraction of patches assigned to context.

    patch_size : int
        Height and width of each square patch.

    device : str or torch.device, optional
        Device for returned indices. Defaults to images.device.

    Returns
    -------
    context_indices, target_indices : torch.Tensor
        Shapes [B, num_context] and [B, num_target].
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")

    batch_size, _, height, width = images.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size")

    if not 0 < context_ratio < 1:
        raise ValueError("context_ratio must be between 0 and 1")

    patches = images.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    # Shape: [B, C, grid_h, grid_w, patch_h, patch_w]

    scores = patches.abs().mean(dim=(1, 4, 5))
    scores = scores.flatten(1)

    num_patches = scores.shape[1]
    num_context = int(num_patches * context_ratio)
    num_target = num_patches - num_context

    context_indices = []
    target_indices = []

    for batch_index in range(batch_size):
        # Randomize ties while preferring higher-content patches.
        noise = torch.rand_like(scores[batch_index]) * 1e-6
        ranked = torch.argsort(scores[batch_index] + noise, descending=True)

        context = ranked[:num_context]
        target = ranked[num_context:]

        context_indices.append(context)
        target_indices.append(target)

    context_indices = torch.stack(context_indices)
    target_indices = torch.stack(target_indices)

    if device is not None:
        context_indices = context_indices.to(device)
        target_indices = target_indices.to(device)

    return context_indices, target_indices
