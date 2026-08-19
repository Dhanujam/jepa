import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Extract 224x224 axial MRI slices."
)

parser.add_argument(
    "input",
    type=str,
    help="Normalized MRI"
)

parser.add_argument(
    "output_dir",
    type=str,
    help="Directory for PNG slices"
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------

input_path = Path(args.input)
output_dir = Path(args.output_dir)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

if not input_path.exists():
    raise FileNotFoundError(
        f"MRI not found: {input_path}"
    )


# --------------------------------------------------
# Subject name
# --------------------------------------------------

subject = input_path.name.split("_T1w")[0]

print("Input:", input_path)


# --------------------------------------------------
# Load MRI
# --------------------------------------------------

img = nib.load(input_path)

volume = img.get_fdata().astype(
    np.float32
)

print("Volume shape:", volume.shape)


# --------------------------------------------------
# Convert normalized values to 0-255
# --------------------------------------------------

CLIP_MIN = -3.0
CLIP_MAX = 3.0

brain_mask = volume != 0

clipped = np.clip(
    volume,
    CLIP_MIN,
    CLIP_MAX
)

scaled = np.zeros_like(
    clipped,
    dtype=np.uint8
)

scaled[brain_mask] = (
    (clipped[brain_mask] - CLIP_MIN)
    / (CLIP_MAX - CLIP_MIN)
    * 255
).astype(np.uint8)


# --------------------------------------------------
# Extract axial slices
# --------------------------------------------------

total = volume.shape[2]

saved = 0
skipped = 0

for index in range(total):

    slice_2d = scaled[:, :, index]

    brain_fraction = (
        np.count_nonzero(slice_2d)
        / slice_2d.size
    )

    if brain_fraction < 0.05:
        skipped += 1
        continue

    slice_2d = np.rot90(slice_2d)

    image = Image.fromarray(slice_2d)

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    output_path = (
        output_dir
        / f"{subject}_T1w_slice_{index:03d}.png"
    )

    image.save(output_path)

    saved += 1


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n--- SLICE EXTRACTION ---")
print("Total:", total)
print("Saved:", saved)
print("Skipped:", skipped)
print("Output:", output_dir)