"""
Training script for the baseline CNN on the CULLPDB dataset.

The training set is split into training and validation at protein level.
The best model, selected on the validation loss, is saved as a checkpoint.
CB513 is never touched here: it is reserved for the final evaluation.

Usage (from the project root)
-----------------------------
    python -m src.train
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    PAD_LABEL,
    RANDOM_SEED,
    TRAIN_FILE,
    ProteinDataset,
    prepare_dataset,
    train_validation_split,
)
from src.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"

BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.2


def create_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Build the training and validation DataLoaders.

    Returns:
        A tuple ``(train_loader, val_loader)``.
    """
    X, y, mask = prepare_dataset(TRAIN_FILE)

    (X_train, y_train, _), (X_val, y_val, _) = train_validation_split(
        X, y, mask,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    train_loader = DataLoader(
        ProteinDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ProteinDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Train the model for one epoch.

    Padding positions do not contribute to the loss, because the criterion
    ignores the ``PAD_LABEL`` target.

    Returns:
        Average training loss over the batches.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)               # (B, 8, 700)
        loss = criterion(logits, y_batch)     # y_batch: (B, 700)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Compute loss and Q8 accuracy on a loader.

    Accuracy is computed on real residues only: padding positions are excluded
    through the mask, otherwise the score would be inflated.

    Returns:
        A tuple ``(average_loss, q8_accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item()

            predictions = logits.argmax(dim=1)    # (B, 700)
            mask = y_batch != PAD_LABEL

            correct += ((predictions == y_batch) & mask).sum().item()
            total += mask.sum().item()

    return total_loss / len(loader), correct / total


def train_all(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    model_name,
):
    """Train the model with early stopping and save the best checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_cnn.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Q8 acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), checkpoint_path)
            print("  Best checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")




def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="baseline",
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    train_loader, val_loader = create_dataloaders()

    model = MODELS[args.model]().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_all(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.model,
    )


if __name__ == "__main__":
    main()

