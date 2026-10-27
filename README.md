# Protein Secondary Structure Prediction (Q8)

Per-residue prediction of protein secondary structure into 8 classes (`L, B, E, G, I, H, S, T`), treated as a sequence labeling task. The project compares a baseline 1D CNN with a proposed residual 1D CNN (PyTorch), trained on CULLPDB and evaluated on CB513 (Zhou & Troyanskaya, ICML 2014).

## Setup

Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/Mac: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the dataset ([MEGA link](https://mega.nz/folder/xct0XSpA#SKz72JtnSAaX61QLMC_JNg)) and place these two files in `data/` (not included in the repository):

- `cullpdb+profile_5926_filtered.npy.gz`: training and validation.
- `cb513+profile_split1.npy.gz`: final test.

Despite the extension, the files may be plain NumPy arrays; the loader handles both formats.

## Data and preprocessing

| Choice | Decision | Motivation |
|---|---|---|
| Raw input | 57 features per residue, padded to 700 positions | Fixed-length sequences enable batching |
| Input features | 46 per residue: 22 amino-acid channels, 2 terminal flags, and 22 sequence-profile features | Combines residue identity and profile information |
| Excluded features | 9 Q8 label channels and 2 solvent-accessibility features | Prevents target leakage and avoids using features unavailable at test time |
| Normalization | No explicit feature normalization is applied | Preserves the original feature representation |
| Padding | Input features are zeroed at padded positions; target label 8 marks padding | Allows padded positions to be excluded from loss and evaluation |
| Train/validation split | CULLPDB filtered (5,365 proteins), randomly split 80/20 at protein level with seed 42: 4,292 training and 1,073 validation proteins | Keeps each protein entirely within one split |
| Test set | CB513, reserved for final evaluation | Avoids using the test set for model selection |
| Class imbalance | No class reweighting or resampling is applied | Provides a baseline for evaluating performance across classes |

The loader accepts plain NumPy arrays and gzip-compressed NumPy files, despite the `.npy.gz` extension. Preprocessing produces input arrays of shape `(N, 700, 46)`, target arrays of shape `(N, 700)`, and a boolean padding mask. The PyTorch dataset converts inputs to channel-first format `(N, 46, 700)` for `Conv1d` models.

The data inspection script reports the class distribution over real residues separately for training and validation, excluding padding positions.

## Models

Both models use 1D convolutions to predict one of 8 secondary-structure classes for each residue.

- **Baseline CNN:** a convolutional layer (46 input channels, 64 output channels, kernel size 5), followed by ReLU, dropout (0.2), and a 1×1 convolution producing 8 logits per residue. Its theoretical receptive field is 5 residues.
- **Proposed Residual CNN:** an input convolution with 128 hidden channels and kernel size 5, followed by two residual blocks. Each block contains two kernel-5 convolutions, ReLU activations, dropout (0.2), and a skip connection. A dropout layer and a 1×1 convolution produce the final 8 logits per residue. Its theoretical receptive field is 21 residues.

The proposed model explicitly derives a padding mask from the amino-acid one-hot channels and applies it after intermediate operations to prevent padded positions from contributing to subsequent convolutions. The baseline does not explicitly apply this mask within its forward pass. For both models, padded target positions are excluded from the loss using `ignore_index=8`, and evaluation metrics are computed on real residues only.

Training uses Adam with a learning rate of 1e-3, a batch size of 8, and a maximum of 30 epochs. Early stopping uses validation loss with a patience of 5 epochs, and the checkpoint with the lowest validation loss is saved. CB513 is reserved for final evaluation and is not used for model selection.

### Evaluation Metrics

All metrics are computed on real amino acid residues only; padding positions are excluded from evaluation.

The evaluation reports:

- **Q8 accuracy:** fraction of correctly classified residues across all eight secondary structure classes.
- **Macro F1:** unweighted mean of the per-class F1 scores, giving equal importance to each class, including rare classes.
- **Weighted F1:** mean of the per-class F1 scores weighted by class support.
- **Per-class precision, recall, and F1:** detailed performance for each Q8 class.
- **Confusion matrix:** compares true and predicted classes. The saved plots use row normalization, so each row represents the distribution of predictions for a given true class.

Classes that are never predicted receive an F1 score of zero. The macro average includes all eight classes, including classes absent from the evaluated labels.

Metrics are saved as JSON files in `outputs/`, while confusion matrix plots are saved in `outputs/figures/`.

## Results
Evaluation saves the metrics as JSON files in `outputs/`, including Q8 accuracy and F1 scores. Confusion matrices are saved in `outputs/figures/`. When evaluating on CB513 with `--split test`, the script also generates up to three qualitative plots per model, showing the ground-truth structure, predictions, and per-residue correctness, under `outputs/figures/{model}/`.

Checkpoints are stored in `outputs/checkpoints/`. The CB513 test set should be used only for final evaluation, not for hyperparameter tuning or model selection.

| Validation set | Baseline CNN | Proposed Residual CNN |
|---|---:|---:|
| Q8 accuracy | 0.6429 | 0.7154 |
| Macro F1 | 0.3451 | 0.4292 |
| Weighted F1 | 0.6100 | 0.6925 |

| Test set (CB513) | Baseline CNN | Proposed Residual CNN |
|---|---:|---:|
| Q8 accuracy | 0.6050 | 0.6762 |
| Macro F1 | 0.3341 | 0.4113 |
| Weighted F1 | 0.5691 | 0.6496 |

The proposed model improves Q8 accuracy on CB513 by 7.1 percentage points. Common classes (`H`, `E`) are predicted well, while rare classes such as `I` remain difficult, which is why macro F1 is much lower than accuracy. For reference, Zhou & Troyanskaya report 66.4% Q8 on CB513; this is not a controlled comparison (different features, data and protocol).

Caveat: the baseline ran for 30 epochs, while the proposed model's training was manually interrupted after 5 epochs. The comparison is therefore not fully controlled, and the proposed model might improve further.

## Usage

```powershell
python -m src.data                                   # inspect preprocessing and class distribution
python -m src.train --model baseline                 # train baseline
python -m src.train --model proposed                 # train proposed model
python -m src.evaluate --model baseline --split val  # validation metrics
python -m src.evaluate --model proposed --split val
python -m src.evaluate --model baseline --split test # CB513 metrics (use only for the final evaluation)
python -m src.evaluate --model proposed --split test
```

Qualitative comparison: evaluating on `--split test` also saves, for selected CB513 proteins, plots comparing the true and predicted secondary structure in `outputs/figures/<model>/qualitative_protein_*.png`. Metrics (`metrics_<model>_<split>.json`) and confusion matrices are saved in `outputs/` and `outputs/figures/`. Checkpoints are in `outputs/checkpoints/`.

## Project structure

```text
src/data.py       loading, preprocessing, split, Dataset
src/models.py     baseline CNN and residual CNN
src/train.py      training with early stopping and checkpointing
src/evaluate.py   metrics, confusion matrix, qualitative plots
src/metrics.py    Q8 accuracy, F1, confusion matrix
```

## Limitations and ideas for improvement

- Controlled comparison: train both models for the same number of epochs, and with several seeds to estimate variance.
- Class imbalance: try a class-weighted loss (e.g. inverse square root of frequency) or focal loss to improve `B`, `G`, `S`, `I`.
- Longer context: dilated convolutions, a BiLSTM/BiGRU on top of the CNN, or a small Transformer encoder.
- Better inputs: embeddings from pretrained protein language models (ESM-2, ProtT5) instead of one-hot and profile.
- Structured decoding: a CRF or smoothing to avoid implausible isolated predictions.
- Evaluation: SOV score, and a homology-aware validation split (the current one is random at protein level).
- Tuning: no systematic hyperparameter search was done, because of the time limit.
- Not implemented: the optional REST API (FastAPI).
