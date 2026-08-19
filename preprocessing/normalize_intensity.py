import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Perform brain-wise intensity normalization."
)

parser.add_argument(
    "input",
    type=str,
    help="Path to skull-stripped brain MRI"
)

parser.add_argument(
    "mask",
    type=str,
    help="Path to HD-BET mask"
)

parser.add_argument(
    "output",
    type=str,
    help="Path to normalized MRI"
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------

input_path = Path(args.input)
mask_path = Path(args.mask)
output_path = Path(args.output)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Check files
# --------------------------------------------------

if not input_path.exists():
    raise FileNotFoundError(
        f"Brain MRI not found: {input_path}"
    )

if not mask_path.exists():
    raise FileNotFoundError(
        f"Brain mask not found: {mask_path}"
    )


print("Brain MRI:", input_path)
print("Brain mask:", mask_path)


# --------------------------------------------------
# Load
# --------------------------------------------------

brain_img = nib.load(input_path)
mask_img = nib.load(mask_path)

brain = brain_img.get_fdata().astype(np.float32)
mask = mask_img.get_fdata() > 0


# --------------------------------------------------
# Check dimensions
# --------------------------------------------------

if brain.shape != mask.shape:
    raise ValueError(
        f"Shape mismatch: "
        f"{brain.shape} vs {mask.shape}"
    )


brain_voxels = brain[mask]

if len(brain_voxels) == 0:
    raise ValueError(
        "Mask contains no brain voxels."
    )


# --------------------------------------------------
# Calculate statistics
# --------------------------------------------------

mean = brain_voxels.mean()
std = brain_voxels.std()

print("\n--- BRAIN STATISTICS ---")
print("Brain voxels:", len(brain_voxels))
print("Mean:", mean)
print("Standard deviation:", std)


# --------------------------------------------------
# Z-score normalization
# --------------------------------------------------

normalized = np.zeros_like(
    brain,
    dtype=np.float32
)

normalized[mask] = (
    (brain[mask] - mean)
    / (std + 1e-8)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

normalized_img = nib.Nifti1Image(
    normalized,
    brain_img.affine,
    brain_img.header
)

nib.save(
    normalized_img,
    output_path
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

values = normalized[mask]

print("\nNormalization completed.")
print("Saved to:", output_path)

print("\n--- NORMALIZED STATISTICS ---")
print("Minimum:", values.min())
print("Maximum:", values.max())
print("Mean:", values.mean())
print("Standard deviation:", values.std())