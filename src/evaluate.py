
"""
Evaluate a trained model on the validation split or CB513 test set.

Save Q8 metrics, confusion matrices and qualitative predictions under
outputs/. Use CB513 only for final evaluation, not model selection.

Usage:
    python -m src.evaluate --split val
    python -m src.evaluate --split test
"""


import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import (
    RANDOM_SEED,
    TEST_FILE,
    TRAIN_FILE,
    ProteinDataset,
    prepare_dataset,
    train_validation_split,
)
from src.metrics import compute_metrics, plot_confusion_matrix, print_report
from src.models import MODELS
from src.train import BATCH_SIZE, CHECKPOINT_DIR, VALIDATION_FRACTION
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_CHECKPOINTS = {
    "baseline": CHECKPOINT_DIR / "baseline_cnn.pt",
    "proposed": CHECKPOINT_DIR / "proposed_cnn.pt",
}

def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for the requested split ("val" or "test")."""
    if split == "test":
        X, y, _ = prepare_dataset(TEST_FILE)
        return X, y

    X, y, mask = prepare_dataset(TRAIN_FILE)
    _, (X_val, y_val, _) = train_validation_split(
        X, y, mask, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED
    )
    return X_val, y_val


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on a loader and return ``(y_true, y_pred)``, shape (N, 700)."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))          # (B, 8, 700)
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y_batch.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)

def plot_qualitative(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    n_samples: int = 3,
) -> None:
    """Plot ground truth, predictions and per-residue correctness for CB513."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Label encoding used by the dataset: 0-7 are Q8 classes, 8 is padding.
    class_names = ["L", "B", "E", "G", "I", "H", "S", "T"]

    class_colors = [
        "#9ECAE1", "#FDD0A2", "#3182BD", "#756BB1",
        "#BCBD22", "#E6550D", "#74C476", "#FD8D3C",
    ]
    class_cmap = ListedColormap(class_colors)
    correctness_cmap = ListedColormap(["#E57373", "#66BB6A"])

    # Select three proteins with at least 100 real residues.
    valid_lengths = np.sum(y_true != 8, axis=1)
    candidates = np.flatnonzero(valid_lengths >= 100)[:n_samples]

    if len(candidates) == 0:
        print("No proteins with at least 100 valid residues found.")
        return

    for idx in candidates:
        # Exclude padding from the visualization.
        mask = y_true[idx] != 8
        true_seq = y_true[idx][mask]
        pred_seq = y_pred[idx][mask]
        correct = (true_seq == pred_seq).astype(int)

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 3.8),
            gridspec_kw={"height_ratios": [1, 1, 0.6]},
        )

        axes[0].imshow(
            true_seq[np.newaxis, :],
            aspect="auto", interpolation="nearest",
            cmap=class_cmap, vmin=0, vmax=7,
        )
        axes[0].set_ylabel("Ground truth")

        axes[1].imshow(
            pred_seq[np.newaxis, :],
            aspect="auto", interpolation="nearest",
            cmap=class_cmap, vmin=0, vmax=7,
        )
        axes[1].set_ylabel("Prediction")

        axes[2].imshow(
            correct[np.newaxis, :],
            aspect="auto", interpolation="nearest",
            cmap=correctness_cmap, vmin=0, vmax=1,
        )
        axes[2].set_ylabel("Correct?")
        axes[2].set_xlabel("Residue position")

        # Show readable position ticks without overcrowding the figure.
        for ax in axes:
            ax.set_yticks([])
            ax.set_xlim(-0.5, len(true_seq) - 0.5)

        tick_positions = np.linspace(
            0, len(true_seq) - 1, min(8, len(true_seq)), dtype=int
        )
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels(tick_positions + 1)

        class_legend = [
            Patch(facecolor=class_colors[i], label=class_names[i])
            for i in range(8)
        ]
        fig.legend(
            handles=class_legend,
            loc="lower center",
            ncol=8,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

        accuracy = correct.mean()
        fig.suptitle(
            f"CB513 protein {idx} — {len(true_seq)} residues "
            f"— accuracy: {accuracy:.1%}"
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.92])

        path = output_dir / f"qualitative_protein_{idx}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved qualitative plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model", choices=list(MODELS), default="baseline")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Train the model first with: "
            f"python -m src.train --model {args.model}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS[args.model]().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    X, y = load_split(args.split)
    loader = DataLoader(ProteinDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)

    name = f"{args.model}_{args.split}"
    print_report(metrics, title=f"Results: {name}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        OUTPUT_DIR / "figures" / f"confusion_matrix_{name}.png",
        title=f"Confusion matrix ({name}, row-normalized)",
    )

    serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    serializable["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(OUTPUT_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved: outputs/metrics_{name}.json and outputs/figures/confusion_matrix_{name}.png")
    

    if args.split == "test":
        plot_qualitative(
            y_true,
            y_pred,
            OUTPUT_DIR / "figures" / args.model,
            n_samples=3
        )

if __name__ == "__main__":
    main()
