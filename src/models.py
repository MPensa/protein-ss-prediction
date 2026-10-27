"""
Models for per-residue Q8 secondary structure prediction.

All models take an input of shape (batch, channels, 700) and return logits of
shape (batch, 8, 700): one score per class for each position.

Padding handling: padding positions are all zeros in the input. Every model
derives the padding mask from the input itself, so that padding never leaks
into the context of real residues. This assumes that padding is only at the
end of each sequence, as in this dataset.
"""

import torch
import torch.nn.functional as F
from torch import nn

# The first 22 input channels are the one-hot amino acid. A real residue always
# has exactly one of them active, while padding positions are all zeros.
AA_CHANNELS = 22


def padding_mask(x: torch.Tensor) -> torch.Tensor:
    """Return a (batch, 1, length) float mask, 1 on real residues and 0 on padding."""
    return (x[:, :AA_CHANNELS].sum(dim=1, keepdim=True) > 0).to(x.dtype)



class BaselineCNN(nn.Module):
    """Simple 1D CNN baseline for per-residue Q8 prediction."""

    def __init__(self, in_channels=46, n_classes=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, n_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x

        h = F.relu(self.conv1(x))
        h = self.dropout(h) * mask
        h = self.conv2(h)

        return F.relu(h + residual) * mask


class ResidualCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 46,
        n_classes: int = 8,
        hidden_channels: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_layer = nn.Conv1d(
            in_channels, hidden_channels,
            kernel_size=5, padding=2,
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dropout=dropout),
            ResidualBlock(hidden_channels, dropout=dropout),
        ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, n_classes, kernel_size=1),
        )

    def forward(self, x):
        mask = padding_mask(x)

        x = F.relu(self.input_layer(x)) * mask

        for block in self.residual_blocks:
            x = block(x, mask)

        return self.classifier(x) * mask


MODELS = {
    "baseline": BaselineCNN,
    "proposed": ResidualCNN,
}

def make_protein(length: int, channels: int = 46) -> torch.Tensor:
    """Random (1, channels, 700) input with ``length`` real residues, then padding."""
    x = torch.zeros(1, channels, 700)
    x[0, :AA_CHANNELS, :length] = torch.eye(AA_CHANNELS)[torch.randint(0, 21, (length,))].T
    x[0, AA_CHANNELS:, :length] = torch.rand(channels - AA_CHANNELS, length)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)

    for name, model_class in MODELS.items():
        model = model_class().eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}: {n_params:,} parameters")

        # Shape check on a batch of three proteins of different lengths.
        batch = torch.cat([make_protein(50), make_protein(300), make_protein(700)])
        with torch.no_grad():
            print("  Input:", tuple(batch.shape), "-> Output:", tuple(model(batch).shape))

        # Padding check: a protein must get the same predictions alone or in a
        # batch with a longer protein (i.e. with a different amount of padding).
        short, long_ = make_protein(100), make_protein(300)
        with torch.no_grad():
            alone = model(short)[..., :100]
            in_batch = model(torch.cat([short, long_]))[0:1, :, :100]

        assert torch.allclose(alone, in_batch, atol=1e-5), f"{name}: padding leaks!"
        print("  Padding check passed.")
