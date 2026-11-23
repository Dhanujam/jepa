import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.classifier import JEPAClassifier


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224


class BrainTumorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(CLASSES):
            class_dir = self.root / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}"
                )

            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp",
                    ".tif", ".tiff", ".webp"
                }:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("L")

            if self.transform:
                image = self.transform(image)

        return image, label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train a JEPA-based brain tumour classifier."
    )

    parser.add_argument(
        "--data-dir",
        default=str(
            Path.home()
            / "Downloads"
            / "brain_tumor_dataset"
            / "Epic and CSCR hospital Dataset"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/jepa_ixi_pretrained.pth",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/jepa_tumor_classifier_weighted.pth",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-fraction", type=float, default=1.0)

    # Test evaluation is optional. Use it only for a final evaluation.
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set after training.",
    )

    parser.add_argument(
        "--random-encoder",
        action="store_true",
        help="Use an untrained encoder instead of loading JEPA weights.",
    )

    parser.add_argument("--split-file", default=None, help="JSON file containing exact train_indices and validation_indices.")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    if not 0 < args.label_fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1].")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Train and Test folders inside {data_dir}"
        )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = BrainTumorDataset(train_dir, transform)

    print("Training images:", len(train_dataset))
    print("Classes:", CLASSES)

    # Split selection: exact external split or original seeded random split.
    if args.split_file:
        split_path = Path(args.split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with split_path.open("r", encoding="utf-8") as handle:
            exact_split = json.load(handle)
        fixed_train_indices = exact_split["train_indices"]
        val_indices = exact_split["validation_indices"]
        if len(fixed_train_indices) != len(set(fixed_train_indices)):
            raise RuntimeError("Split train_indices contains duplicates")
        if len(val_indices) != len(set(val_indices)):
            raise RuntimeError("Split validation_indices contains duplicates")
        if set(fixed_train_indices) & set(val_indices):
            raise RuntimeError("Train and validation indices overlap")
        fixed_train_pool = torch.utils.data.Subset(train_dataset, fixed_train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        print("Loaded exact split:", split_path)
        print("Exact train/validation counts:", len(fixed_train_pool), len(val_subset))
    else:
        generator = torch.Generator().manual_seed(args.seed)
        fixed_train_pool, val_subset = torch.utils.data.random_split(
            train_dataset, [0.9, 0.1], generator=generator
        )

    if args.label_fraction < 1.0:
        labelled_indices = []
        rng = np.random.default_rng(args.seed)
        # Resolve dataset indices for exact split subsets and random subsets alike.
        pool_indices = list(fixed_train_pool.indices) if isinstance(fixed_train_pool, torch.utils.data.Subset) else list(fixed_train_pool.indices)
        for class_index in range(len(CLASSES)):
            class_indices = [
                dataset_index for dataset_index in pool_indices
                if train_dataset.samples[dataset_index][1] == class_index
            ]
            count = max(1, int(round(len(class_indices) * args.label_fraction)))
            labelled_indices.extend(rng.choice(class_indices, size=count, replace=False).tolist())
        training_pool = torch.utils.data.Subset(train_dataset, labelled_indices)
    elif isinstance(fixed_train_pool, torch.utils.data.Subset):
        training_pool = fixed_train_pool
    else:
        training_pool = fixed_train_pool

    print("Labelled training pool:", len(training_pool))

    train_subset = training_pool
    # Calculate class weights using only the sampled training subset.
    train_labels = []
    for subset_index in training_pool.indices:
        train_labels.append(int(train_dataset.samples[subset_index][1]))

    class_counts = torch.bincount(
        torch.tensor(train_labels, dtype=torch.long),
        minlength=len(CLASSES),
    ).float()

    if (class_counts == 0).any():
        raise ValueError(
            "At least one class has no training examples."
        )

    class_weights = class_counts.sum() / (
        len(CLASSES) * class_counts
    )

    class_weights = class_weights.to(device)

    print("\nTraining subset class counts:")
    for class_name, count, weight in zip(
        CLASSES, class_counts.tolist(), class_weights.tolist()
    ):
        print(
            f"  {class_name}: {int(count)} images, "
            f"weight={weight:.4f}"
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    torch.use_deterministic_algorithms(False)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load the JEPA-pretrained encoder.
    model = JEPAClassifier(
        num_classes=len(CLASSES),
        freeze_encoder=True,
    )

    checkpoint_path = Path(args.checkpoint)

    if args.random_encoder:
        print("\nUsing randomly initialized encoder (no JEPA checkpoint loaded).")
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"JEPA checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        result = model.encoder.load_state_dict(
            checkpoint["context_encoder"],
            strict=True,
        )

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Encoder checkpoint keys do not match.")

        print("\nLoaded JEPA checkpoint:", checkpoint_path)
        print(
            "JEPA pretraining steps:",
            checkpoint.get("step", "unknown"),
        )

    model = model.to(device)

    # Weighted loss helps address imbalanced class learning.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_accuracy = -1.0
    best_val_macro_f1 = None
    best_val_recalls = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        # Validate after every epoch.
        model.eval()

        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu()

                val_predictions.extend(predictions.tolist())
                val_labels.extend(labels.tolist())

        train_loss = total_loss / total
        train_acc = correct / total

        val_acc = sum(
            pred == label
            for pred, label in zip(val_predictions, val_labels)
        ) / len(val_labels)

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        # Show class-wise validation results.
        report = classification_report(
            val_labels,
            val_predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        print("Validation recall by class:")
        for class_name in CLASSES:
            print(
                f"  {class_name}: "
                f"{report[class_name]['recall']:.4f}"
            )

        print(
            f"  Macro F1: "
            f"{report['macro avg']['f1-score']:.4f}"
        )

        # Keep the model with the best validation accuracy.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_macro_f1 = report["macro avg"]["f1-score"]
            best_val_recalls = {
                name: report[name]["recall"] for name in CLASSES
            }
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best model was produced.")

    # Save the selected model without touching the JEPA checkpoint
    # or the original baseline classifier checkpoint.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "classes": CLASSES,
            "class_to_index": {name: index for index, name in enumerate(CLASSES)},
            "image_size": IMAGE_SIZE,
            "best_validation_accuracy": best_val_accuracy,
            "pretraining_steps": 0 if args.random_encoder else checkpoint.get("step"),
            "encoder_initialization": "random" if args.random_encoder else "jepa",
            "source_checkpoint": None if args.random_encoder else str(checkpoint_path),
            "seed": args.seed,
            "validation_fraction": 0.1,
            "preprocessing": "resize_224_rgb_to_grayscale_to_tensor",
            "class_weights": class_weights.cpu(),
            "label_fraction": args.label_fraction,
            "labelled_sample_count": len(training_pool),
            "train_sample_count": len(training_pool),
            "validation_sample_count": len(val_subset),
            "best_validation_macro_f1": best_val_macro_f1,
            "best_validation_recall_by_class": best_val_recalls,
        },
        output_path,
    )

    print("\nBest validation accuracy:", f"{best_val_accuracy:.4f}")
    print("Saved weighted classifier:", output_path)

    # Evaluate the held-out test set only when explicitly requested.
    if args.evaluate_test:
        test_dataset = BrainTumorDataset(test_dir, transform)

        print("\nTest images:", len(test_dataset))

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model.load_state_dict(best_state)
        model.eval()

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu()

                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        test_accuracy = sum(
            pred == label
            for pred, label in zip(all_predictions, all_labels)
        ) / len(all_labels)

        print("\nTest accuracy:", f"{test_accuracy:.4f}")
        print("\nTest classification report:")

        print(
            classification_report(
                all_labels,
                all_predictions,
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                digits=4,
                zero_division=0,
            )
        )

        print(
            "Test confusion matrix "
            "(rows=true, columns=predicted):"
        )

        print(
            confusion_matrix(
                all_labels,
                all_predictions,
                labels=list(range(len(CLASSES))),
            )
        )


if __name__ == "__main__":
    main()
