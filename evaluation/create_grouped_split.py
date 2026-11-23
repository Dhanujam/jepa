import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_classifier import BrainTumorDataset, CLASSES, IMAGE_SIZE

DATA_ROOT = Path(r"C:\Users\admin\Downloads\brain_tumor_dataset\Epic and CSCR hospital Dataset")
TRAIN_ROOT = DATA_ROOT / "Train"
OUTPUT_DIR = Path("evaluation/group_aware_split_seed42")
SEED = 42
VAL_FRACTION = 0.10


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    dataset = BrainTumorDataset(TRAIN_ROOT)
    hash_groups = defaultdict(list)
    for index, (path, label) in enumerate(dataset.samples):
        hash_groups[sha256(path)].append(index)

    # Each hash group is assigned entirely to train or validation.
    # Class-labelled groups are treated as units for proportional assignment.
    groups = []
    for image_hash, indices in hash_groups.items():
        labels = {dataset.samples[index][1] for index in indices}
        if len(labels) != 1:
            raise RuntimeError(f"Duplicate group {image_hash} has conflicting labels")
        label = next(iter(labels))
        class_name = CLASSES[label]
        groups.append({"hash": image_hash, "indices": indices, "label": label, "class_name": class_name})

    rng = random.Random(SEED)
    rng.shuffle(groups)

    target_validation_files = round(len(dataset) * VAL_FRACTION)
    train_indices = []
    val_indices = []
    val_classes = Counter()

    # Greedy proportional assignment: select whole groups by size and
    # nearest count to each class's 10% target, with deterministic tie-breaking.
    for label in range(len(CLASSES)):
        class_groups = [g for g in groups if g["label"] == label]
        class_groups.sort(key=lambda g: (len(g["indices"]), g["hash"]), reverse=True)
        class_total = sum(len(g["indices"]) for g in class_groups)
        class_target = round(class_total * VAL_FRACTION)
        assigned = 0
        for group in class_groups:
            if assigned < class_target and assigned + len(group["indices"]) <= class_target + max(1, len(group["indices"])):
                val_indices.extend(group["indices"])
                val_classes[label] += len(group["indices"])
                assigned += len(group["indices"])
            else:
                train_indices.extend(group["indices"])
        # Keep a group-aware fallback if rounding left any class short/over.
        while assigned < class_target:
            candidates = [g for g in class_groups if all(i not in val_indices for i in g["indices"])]
            if not candidates:
                break
            candidates.sort(key=lambda g: abs((assigned + len(g["indices"])) - class_target))
            group = candidates[0]
            val_indices.extend(group["indices"])
            val_classes[label] += len(group["indices"])
            assigned += len(group["indices"])

    selected_val = set(val_indices)
    train_indices = [i for i in range(len(dataset)) if i not in selected_val]

    # Remove conflicting duplicates and confirm every hash group is contained by one side.
    train_hashes = {sha256(dataset.samples[i][0]) for i in train_indices}
    val_hashes = {sha256(dataset.samples[i][0]) for i in val_indices}
    overlap_hashes = train_hashes & val_hashes
    if overlap_hashes:
        raise RuntimeError(f"Group-aware split has {len(overlap_hashes)} overlapping hashes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "validation_fraction_requested": VAL_FRACTION,
        "image_size": IMAGE_SIZE,
        "classes": CLASSES,
        "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        "total_training_files": len(dataset),
        "unique_hash_groups": len(hash_groups),
        "train_indices": sorted(train_indices),
        "validation_indices": sorted(val_indices),
        "train_file_count": len(train_indices),
        "validation_file_count": len(val_indices),
        "train_class_counts": [sum(dataset.samples[i][1] == label for i in train_indices) for label in range(len(CLASSES))],
        "validation_class_counts": [sum(dataset.samples[i][1] == label for i in val_indices) for label in range(len(CLASSES))],
        "overlapping_hash_count": len(overlap_hashes),
        "test_folder_used": False,
    }
    (OUTPUT_DIR / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Saved", OUTPUT_DIR / "split.json")


if __name__ == "__main__":
    main()
