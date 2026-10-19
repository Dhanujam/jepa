import matplotlib.pyplot as plt


def plot_label_efficiency(label_percentages, accuracies, output_path="label_efficiency.png"):
    """Plot measured accuracy against labelled training data percentage."""
    if len(label_percentages) != len(accuracies):
        raise ValueError("Percentages and accuracies must have the same length.")
    if not label_percentages:
        raise ValueError("Provide at least one percentage and accuracy.")
    if any(not 0 < p <= 100 for p in label_percentages):
        raise ValueError("Label percentages must be greater than 0 and at most 100.")
    if any(not 0 <= a <= 1 for a in accuracies):
        raise ValueError("Accuracies must be between 0 and 1.")

    plt.figure(figsize=(7, 5))
    plt.plot(label_percentages, accuracies, marker="o")
    plt.xlabel("Labelled training data (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Labelled Data Percentage")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")
