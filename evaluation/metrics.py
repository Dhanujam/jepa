
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(y_true, y_pred, class_names=None):
    """Calculate classification metrics for a trained model."""
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    labels = list(range(len(class_names))) if class_names else None

    matrix = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ) if class_names else classification_report(
        y_true, y_pred, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix,
        "classification_report": report,
    }
