import argparse
from pathlib import Path

import SimpleITK as sitk


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Perform N4 bias-field correction on an MRI."
)

parser.add_argument(
    "input",
    type=str,
    help="Path to input canonical NIfTI MRI"
)

parser.add_argument(
    "output",
    type=str,
    help="Path to output N4-corrected MRI"
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
# Read MRI
# --------------------------------------------------

image = sitk.ReadImage(
    str(input_path)
)

image = sitk.Cast(
    image,
    sitk.sitkFloat32
)

print(
    "Image size:",
    image.GetSize()
)

print(
    "Image spacing:",
    image.GetSpacing()
)


# --------------------------------------------------
# Create foreground mask
# --------------------------------------------------

print("Creating foreground mask...")

mask = sitk.OtsuThreshold(
    image,
    0,
    1,
    200
)

mask = sitk.Cast(
    mask,
    sitk.sitkUInt8
)


# --------------------------------------------------
# N4 bias-field correction
# --------------------------------------------------

print("Running N4 bias correction...")

corrector = (
    sitk.N4BiasFieldCorrectionImageFilter()
)

corrector.SetMaximumNumberOfIterations(
    [50, 50, 30, 20]
)

corrected = corrector.Execute(
    image,
    mask
)


# --------------------------------------------------
# Save
# --------------------------------------------------

sitk.WriteImage(
    corrected,
    str(output_path)
)

print("\nN4 correction completed.")

print(
    f"Saved to: {output_path}"
)