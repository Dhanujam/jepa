from pathlib import Path
import argparse

import nibabel as nib


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Convert an MRI to canonical orientation."
)

parser.add_argument(
    "input",
    type=str,
    help="Path to input NIfTI MRI"
)

parser.add_argument(
    "output",
    type=str,
    help="Path to output NIfTI MRI"
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------

input_path = Path(args.input)
output_path = Path(args.output)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not input_path.exists():
    raise FileNotFoundError(
        f"Input MRI not found: {input_path}"
    )

print(f"Input MRI: {input_path}")


# --------------------------------------------------
# Load MRI
# --------------------------------------------------

img = nib.load(input_path)

print(
    "Original orientation:",
    nib.aff2axcodes(img.affine)
)

print(
    "Original shape:",
    img.shape
)


# --------------------------------------------------
# Convert to canonical orientation
# --------------------------------------------------

canonical_img = nib.as_closest_canonical(img)

print(
    "Canonical orientation:",
    nib.aff2axcodes(canonical_img.affine)
)

print(
    "Canonical shape:",
    canonical_img.shape
)


# --------------------------------------------------
# Save
# --------------------------------------------------

nib.save(
    canonical_img,
    output_path
)

print("\nSaved standardized MRI to:")
print(output_path)