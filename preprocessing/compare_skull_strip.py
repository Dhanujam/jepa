from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


N4_DIR = Path("datasets/processed/ixi/n4")
BET_DIR = Path("datasets/processed/ixi/skull_stripped")

n4_file = list(N4_DIR.glob("*.nii.gz"))[0]

# Find the brain-extracted output.
brain_files = [
    p for p in BET_DIR.glob("*.nii.gz")
    if "mask" not in p.name.lower()
    and "bet" not in p.name.lower()
]

if not brain_files:
    raise FileNotFoundError("Brain-extracted image not found.")

brain_file = brain_files[0]

n4_img = nib.load(n4_file)
brain_img = nib.load(brain_file)

n4 = n4_img.get_fdata()
brain = brain_img.get_fdata()

slice_index = n4.shape[2] // 2

# Before skull stripping
plt.figure(figsize=(6, 6))
plt.imshow(np.rot90(n4[:, :, slice_index]), cmap="gray")
plt.title("Before Skull Stripping")
plt.axis("off")
plt.show()

# After skull stripping
plt.figure(figsize=(6, 6))
plt.imshow(np.rot90(brain[:, :, slice_index]), cmap="gray")
plt.title("After Skull Stripping")
plt.axis("off")
plt.show()