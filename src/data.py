
"""
Data loading and preprocessing for Q8 protein secondary structure prediction.

Prepares CULLPDB and CB513 data as:
    X:    (N, 700, 46) float32 input features
    y:    (N, 700)     labels 0-7; 8 denotes padding
    mask: (N, 700)     True for real residues, False for padding

Inputs include amino acid one-hot encoding (22), terminal flags (2),
and sequence profiles (22). Target labels and solvent accessibility are
excluded to prevent target leakage and ensure test-time availability.

Training/validation splitting is performed at protein level. CB513 is
reserved for final evaluation.

Usage
-----
    python -m src.data        # inspects the data and prints the class distribution
"""

import gzip
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_FILE = "cullpdb+profile_5926_filtered.npy.gz"
TEST_FILE = "cb513+profile_split1.npy.gz"

AMINO_ACIDS = [
    "A", "C", "E", "D", "G", "F", "I", "H", "K", "M",
    "L", "N", "Q", "P", "S", "R", "T", "W", "V", "Y",
    "X", "NoSeq",
]

SECONDARY_STRUCTURES = ["L", "B", "E", "G", "I", "H", "S", "T", "NoSeq"]

SEQ_LEN = 700
N_FEATURES = 57
N_CLASSES = 8          # real Q8 classes, NoSeq excluded
PAD_AMINO_ACID = 21    # index of NoSeq among the amino acid features
PAD_LABEL = 8          # index of NoSeq among the secondary structure labels
RANDOM_SEED = 42


import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    """PyTorch dataset returning one protein per item.

    Args:
        X: Input features, shape (N, 700, F), as returned by ``prepare_dataset``.
        y: Labels, shape (N, 700), with ``PAD_LABEL`` on padding positions.

    Each item is a tuple ``(x, y)``:
        - x: float32 tensor of shape (F, 700), channels first;
        - y: int64 tensor of shape (700,).

    The padding mask is not stored: it can be recovered with ``y != PAD_LABEL``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")

        # Inputs must be float32; CrossEntropyLoss needs int64 (long) class indices.
        # as_tensor does not copy the data when the dtype already matches.
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

        # Conv1d expects (batch, channels, length), so move the features to the
        # channel axis: (N, 700, F) -> (N, F, 700). permute returns a view,
        # so it uses no additional memory.
        self.X = self.X.permute(0, 2, 1)

    def __len__(self) -> int:
        """Return the number of proteins."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the features (F, 700) and the labels (700,) of one protein."""
        return self.X[idx], self.y[idx]
    

def load_dataset(path: Path) -> np.ndarray:
    """Load a dataset file and return it with shape (N_proteins, 700, 57).

    The files are named ``.npy.gz`` but may be either plain NumPy arrays or
    gzip-compressed ones, so both formats are tried. A flat (N, 700 * 57)
    matrix is reshaped automatically.

    Args:
        path: Path to the dataset file.

    Returns:
        Array of shape (N_proteins, 700, 57).

    Raises:
        ValueError: If the array does not have the expected shape.
    """
    # First try to read the file as a plain NumPy array.
    try:
        data = np.load(path, allow_pickle=False)

        if not isinstance(data, np.ndarray):
            raise ValueError("The file does not contain a NumPy array.")

        print("Detected format: plain NumPy array")

    except (ValueError, OSError):
        # Not a valid .npy file: try to read it as gzip.
        with gzip.open(path, "rb") as f:
            data = np.load(f, allow_pickle=False)

        print("Detected format: gzip-compressed NumPy array")

    if data.ndim == 2 and data.shape[1] == SEQ_LEN * N_FEATURES:
        data = data.reshape(-1, SEQ_LEN, N_FEATURES)

    if data.ndim != 3 or data.shape[1:] != (SEQ_LEN, N_FEATURES):
        raise ValueError(f"Unexpected shape: {data.shape}")

    return data


def inspect_dataset(filename: str) -> None:
    """Print a report on the structure and quality of a dataset file.

    The report includes shape, dtype, NaN/inf checks, a decoded preview of the
    first protein, class counts, one-hot consistency checks, and the values taken
    by a few feature columns on real residues and on padding.

    Args:
        filename: Name of the file inside the data directory.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    data = load_dataset(path)

    print("Shape:", data.shape)
    print("Value dtype:", data.dtype)
    print("NaN values:", np.isnan(data).sum())
    print("All values finite:", np.isfinite(data).all())

    # First protein, first 5 positions.
    protein = data[0]
    print("\nFirst 5 positions, first 10 features:")
    print(protein[:5, :10])

    # Decode the one-hot features, following the dataset README.
    print("\nFirst 20 positions:")
    print("Pos | Amino acid | True structure")

    aa_features = protein[:, 0:22]
    ss_features = protein[:, 22:31]

    for i in range(20):
        aa = AMINO_ACIDS[int(np.argmax(aa_features[i]))]
        ss = SECONDARY_STRUCTURES[int(np.argmax(ss_features[i]))]
        print(f"{i:3} | {aa:10} | {ss}")

    # Class counts, excluding padding.
    label_indices = np.argmax(data[:, :, 22:31], axis=2)
    aa_indices = np.argmax(data[:, :, 0:22], axis=2)

    valid = (aa_indices != PAD_AMINO_ACID) & (label_indices != PAD_LABEL)

    print("\nNumber of proteins:", data.shape[0])
    print("Valid positions:", int(valid.sum()))
    print("Padding positions:", int((~valid).sum()))

    print("\nCount of real structures:")
    for i, label in enumerate(SECONDARY_STRUCTURES[:N_CLASSES]):
        count = int(((label_indices == i) & valid).sum())
        print(f"{label}: {count}")

    print("NoSeq label count:", int((label_indices == PAD_LABEL).sum()))

    # One-hot encoding checks.
    aa = data[:, :, 0:22]
    ss = data[:, :, 22:31]

    print("\nEncoding checks:")
    print(
        "Positions whose amino acid features do not sum to 1:",
        int((~np.isclose(aa.sum(axis=2), 1)).sum()),
    )
    print(
        "Positions whose label features do not sum to 1:",
        int((~np.isclose(ss.sum(axis=2), 1)).sum()),
    )
    print(
        "Mismatches between amino acid padding and label padding:",
        int(((aa_indices == PAD_AMINO_ACID) != (label_indices == PAD_LABEL)).sum()),
    )

    # Values of the terminal flags (31, 32), solvent accessibility (33, 34)
    # and the last profile column (56).
    columns = [31, 32, 33, 34, 56]
    print(f"\nValues of feature columns {columns}:")

    for i in columns:
        values = data[:, :, i]

        print(f"\nFeature {i}")
        print("Values on valid residues:", np.unique(values[valid])[:20])
        print("Values on padding:", np.unique(values[~valid])[:20])


def prepare_dataset(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build model inputs, targets and padding mask from a dataset file.

    Inputs are amino acids (22), terminal flags (2) and sequence profile (22),
    for a total of 46 features. Labels and solvent accessibility are excluded.
    All input features are set to zero on padding positions.

    Args:
        filename: Name of the file inside the data directory.

    Returns:
        A tuple ``(X, y, mask)`` where:
            - X has shape (N, 700, 46) and dtype float32;
            - y has shape (N, 700) with classes 0-7, and 8 on padding;
            - mask has shape (N, 700), True on real residues.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    raw_data = load_dataset(path)

    data = np.concatenate(
        [
            raw_data[:, :, 0:22],    # amino acids
            raw_data[:, :, 31:33],   # terminal flags
            raw_data[:, :, 35:57],   # sequence profile
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    labels = np.argmax(raw_data[:, :, 22:31], axis=-1)
    mask = labels != PAD_LABEL

    # Set every feature on padding positions to zero.
    data[~mask] = 0

    print("Original shape:", raw_data.shape)
    print("Shape after feature selection:", data.shape)
    print("Labels shape:", labels.shape)
    print("Mask shape:", mask.shape)
    print("Valid positions:", int(mask.sum()))
    print("Padding positions:", int((~mask).sum()))

    del raw_data  # free memory: the raw array can be large

    return data, labels, mask


def train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    validation_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Split the data into training and validation sets at protein level.

    Each protein belongs entirely to one of the two sets. The split is random
    but reproducible thanks to the fixed seed.

    Args:
        X: Input features, shape (N, 700, F).
        y: Labels, shape (N, 700).
        mask: Padding mask, shape (N, 700).
        validation_fraction: Fraction of proteins assigned to validation.
        seed: Random seed for reproducibility.

    Returns:
        Two tuples, ``(X_train, y_train, mask_train)`` and
        ``(X_val, y_val, mask_val)``.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    protein_indices = np.arange(X.shape[0])

    train_idx, val_idx = train_test_split(
        protein_indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )

    print("Training proteins:", len(train_idx))
    print("Validation proteins:", len(val_idx))
    print("Shared indices:", len(np.intersect1d(train_idx, val_idx)))

    train = (X[train_idx], y[train_idx], mask[train_idx])
    val = (X[val_idx], y[val_idx], mask[val_idx])

    return train, val


def inspect_class_distribution(y: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    """Print the class distribution over real residues (padding excluded).

    Args:
        y: Labels, shape (N, 700).
        mask: Padding mask, shape (N, 700).
        name: Name of the subset, used in the printed header.

    Returns:
        Array of shape (8,) with the number of residues of each class.
    """
    valid_labels = y[mask]

    print(f"\nClass distribution - {name}")
    print(f"Total real residues: {len(valid_labels)}")

    counts = np.bincount(valid_labels, minlength=N_CLASSES)

    for class_id, class_name in enumerate(SECONDARY_STRUCTURES[:N_CLASSES]):
        count = int(counts[class_id])
        percentage = count / len(valid_labels) * 100
        print(f"{class_name}: {count:>8} residues ({percentage:5.2f}%)")

    return counts


def main() -> None:
    """Prepare the training data, split it and print class distributions."""
    X, y, mask = prepare_dataset(TRAIN_FILE)

    (X_train, y_train, mask_train), (X_val, y_val, mask_val) = train_validation_split(
        X, y, mask
    )

    inspect_class_distribution(y_train, mask_train, "Training set")
    inspect_class_distribution(y_val, mask_val, "Validation set")

    print("\nX_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("mask_train:", mask_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("mask_val:", mask_val.shape)


if __name__ == "__main__":
    main()