import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.classifier import JEPAClassifier
from training.train_classifier import BrainTumorDataset, IMAGE_SIZE
from torchvision import transforms

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved NeuroJEPA classifier on real images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="Folder containing a test split or a single class-organized directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-prefix", default="evaluation/results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes", CLASSES)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    dataset = BrainTumorDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAClassifier(num_classes=len(classes), freeze_encoder=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())

    report = classification_report(
        labels_all, predictions_all, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels_all, predictions_all, labels=list(range(len(classes))))
    print("Accuracy:", accuracy_score(labels_all, predictions_all))
    print("Macro F1:", report["macro avg"]["f1-score"])
    print(classification_report(labels_all, predictions_all, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": classes, "accuracy": accuracy_score(labels_all, predictions_all), "macro_f1": report["macro avg"]["f1-score"], "report": report, "confusion_matrix": cm.tolist(), "labels": labels_all, "predictions": predictions_all}, output.with_suffix(".pth"))
    print("Saved metrics to", output.with_suffix(".pth"))


if __name__ == "__main__":
    main()
