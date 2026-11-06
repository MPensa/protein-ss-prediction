import numpy as np
from src.data import load_dataset, DATA_DIR, TRAIN_FILE

raw = load_dataset(DATA_DIR / TRAIN_FILE)
for i in range(3):
    length = int((raw[i, :, 30] == 0).sum())
    print(np.flatnonzero(raw[i, :, 31]), np.flatnonzero(raw[i, :, 32]), length)