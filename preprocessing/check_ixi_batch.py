from pathlib import Path

import nibabel as nib


INPUT_DIR = Path("datasets/raw/ixi")

files = sorted(INPUT_DIR.glob("*_T1w.nii"))


print(f"Found {len(files)} T1 MRI files.\n")


for file in files:

    print("=" * 50)
    print("File:", file.name)

    try:
        image = nib.load(file)

        print("Shape:", image.shape)
        print("Spacing:", image.header.get_zooms()[:3])
        print("Orientation:", nib.aff2axcodes(image.affine))

        print("Status: OK")

    except Exception as error:

        print("Status: FAILED")
        print("Error:", error)
        