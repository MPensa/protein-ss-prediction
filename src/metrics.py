"""
Evaluation metrics for per-residue Q8 secondary structure prediction.

All metrics are computed on real residues only: positions whose true label is
``PAD_LABEL`` (padding) are discarded before counting. Otherwise the score would
be inflated, since about 70% of the positions in the dataset are padding.

Metrics
-------
- Q8 accuracy: fraction of residues classified correctly. It is the metric used
  in the reference paper, but it is dominated by the frequent classes (H, E, L).
- Per-class precision, recall and F1: show how each structure is handled.
- Macro F1: unweighted mean of the per-class F1. Every class counts the same, so
  rare classes such as I and B weigh as much as H. This is the main metric to
  detect a model that ignores the rare classes.
- Weighted F1: mean of the per-class F1 weighted by class frequency. It reflects
  overall performance, like accuracy, while being more informative.
- Confusion matrix: shows which structures are confused with each other.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import N_CLASSES, PAD_LABEL, SECONDARY_STRUCTURES

CLASS_NAMES = SECONDARY_STRUCTURES[:N_CLASSES]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Q8 accuracy, F1 scores and the confusion matrix.

    Args:
        y_true: True labels, any shape, with ``PAD_LABEL`` on padding positions.
        y_pred: Predicted classes (0-7), same shape as ``y_true``.

    Returns:
        A dictionary with:
            - ``q8_accuracy``, ``macro_f1``, ``weighted_f1``: floats;
            - ``per_class``: {class name: precision, recall, f1, support};
            - ``confusion_matrix``: (8, 8) array of counts, rows = true class,
              columns = predicted class.

    Note:
        A class that is never predicted gets F1 = 0 (``zero_division=0``), and
        a class absent from ``y_true`` also counts as 0 in the macro average.
    """
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
    """Save the confusion matrix as a PNG image.

    Args:
        cm: (8, 8) confusion matrix of counts.
        path: Output file path.
        normalize: If True, show row-normalized values, so that rare classes
            remain readable; otherwise show raw counts.
        title: Plot title.
    """
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

    # Write the value in each cell, with a readable text color.
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
