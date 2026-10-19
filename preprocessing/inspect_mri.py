from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Location of our raw IXI MRI
DATA_DIR = Path("datasets/raw/ixi")

# Find NIfTI files
files = list(DATA_DIR.glob("*.nii")) + list(DATA_DIR.glob("*.nii.gz"))

if not files:
    raise FileNotFoundError(
        f"No NIfTI files found in {DATA_DIR}"
    )

# Use the first MRI for our initial test
mri_path = files[0]

print(f"Loading MRI: {mri_path}")

# Load MRI
img = nib.load(mri_path)
print("Orientation:", nib.aff2axcodes(img.affine))
# Convert MRI data to NumPy array
volume = img.get_fdata()

# Display information
print("\n--- MRI INFORMATION ---")
print("File:", mri_path.name)
print("Shape:", volume.shape)
print("Voxel spacing:", img.header.get_zooms())
print("Data type:", volume.dtype)
print("Minimum intensity:", volume.min())
print("Maximum intensity:", volume.max())
print("Mean intensity:", volume.mean())
print("Standard deviation:", volume.std())

# Select middle axial slice
slice_index = volume.shape[2] // 2
slice_image = volume[:, :, slice_index]

# Display slice
plt.figure(figsize=(6, 6))
plt.imshow(
    np.rot90(slice_image),
    cmap="gray"
)

plt.title(f"{mri_path.name} - Axial Slice {slice_index}")
plt.axis("off")
plt.show()