
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path="confusion_matrix.png",
):
    """Plot and save a confusion matrix."""
    labels = list(range(len(class_names)))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")
