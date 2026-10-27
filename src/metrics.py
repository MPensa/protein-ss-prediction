
"""
Evaluation metrics for Q8 protein secondary structure prediction.

Metrics are computed on real residues only, excluding positions labeled
PAD_LABEL. Reports include accuracy, per-class precision/recall/F1,
macro F1, weighted F1 and the confusion matrix.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import N_CLASSES, PAD_LABEL, SECONDARY_STRUCTURES

CLASS_NAMES = SECONDARY_STRUCTURES[:N_CLASSES]



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Q8 accuracy, F1 scores and confusion matrix, excluding padding."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    # Drop padding positions.
    valid = y_true != PAD_LABEL
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    labels = list(range(N_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return {
        "q8_accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support / support.sum()).sum()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1 (row i = how true class i is classified)."""
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)  # avoid division by zero
    return cm / row_sums


def print_report(metrics: dict, title: str = "") -> None:
    """Print a readable summary of the metrics returned by ``compute_metrics``."""
    if title:
        print(f"\n{title}")
    print(f"Q8 accuracy : {metrics['q8_accuracy']:.4f}")
    print(f"Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 : {metrics['weighted_f1']:.4f}")

    print(f"\n{'Class':<6}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>12}")
    for name, m in metrics["per_class"].items():
        print(
            f"{name:<6}{m['precision']:>10.3f}{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}{m['support']:>12d}"
        )



def plot_confusion_matrix(
    cm: np.ndarray,
    path: Path,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> None:
    """Save a raw or row-normalized confusion matrix as a PNG."""

    values = normalize_confusion_matrix(cm) if normalize else cm

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1.0 if normalize else None)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = values.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
