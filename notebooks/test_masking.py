import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.masking import create_random_masks


def main():

    batch_size = 2
    num_patches = 196

    context_indices, target_indices = create_random_masks(
        batch_size=batch_size,
        num_patches=num_patches,
        context_ratio=0.75
    )

    print("Context shape:", context_indices.shape)
    print("Target shape :", target_indices.shape)

    print("\nContext patches for image 1:")
    print(context_indices[0])

    print("\nTarget patches for image 1:")
    print(target_indices[0])


if __name__ == "__main__":
    main()