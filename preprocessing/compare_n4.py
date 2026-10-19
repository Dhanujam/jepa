from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Find files
orientation_dir = Path("datasets/processed/ixi/orientation")
n4_dir = Path("datasets/processed/ixi/n4")

original_file = list(orientation_dir.glob("*.nii.gz"))[0]
n4_file = list(n4_dir.glob("*.nii.gz"))[0]

# Load
original = nib.load(original_file).get_fdata()
corrected = nib.load(n4_file).get_fdata()

# Use the middle slice
slice_index = original.shape[2] // 2

original_slice = original[:, :, slice_index]
corrected_slice = corrected[:, :, slice_index]

# Plot original
plt.figure(figsize=(6, 6))
plt.imshow(np.rot90(original_slice), cmap="gray")
plt.title("Before N4 Bias Correction")
plt.axis("off")
plt.show()

# Plot corrected
plt.figure(figsize=(6, 6))
plt.imshow(np.rot90(corrected_slice), cmap="gray")
plt.title("After N4 Bias Correction")
plt.axis("off")
plt.show()