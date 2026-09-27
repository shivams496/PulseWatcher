# src/split_holdout.py
#
# Fixes the train/eval leakage in the original pipeline:
#   - train.py was early-stopping on data/test.npy
#   - evaluate.py was then tuning the threshold AND reporting final
#     metrics on that same data/test.npy + data/anomaly.npy
#   -> the reported numbers were biased, since the "test" set had
#      already influenced both model selection and threshold choice.
#
# This script splits the EXISTING test.npy / anomaly.npy files
# (no need to redownload MIT-BIH) into:
#   - val_normal.npy   / val_anomaly.npy    -> used ONLY for early
#     stopping (train.py) and threshold selection (evaluate.py)
#   - test_normal.npy  / test_anomaly.npy   -> touched exactly once,
#     only to compute the final reported metrics
#
# Run this ONCE, after preprocess.py and before train.py / evaluate.py:
#   python -m src.split_holdout

import numpy as np
import os

DATA_DIR = "data"
SEED = 42
VAL_FRACTION = 0.5  # split the existing held-out pool 50/50


def split_and_save(source_file, val_name, test_name, val_fraction=VAL_FRACTION):
    path = os.path.join(DATA_DIR, source_file)
    data = np.load(path, allow_pickle=True)
    # Some numpy versions round-trip these as dtype=object; force back to a
    # plain float32 2D array (safe here since preprocess.py already enforced
    # a fixed window size, so every row is the same length).
    data = np.asarray(data, dtype=np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(data))
    split = int(len(data) * val_fraction)
    val_idx, test_idx = idx[:split], idx[split:]

    np.save(os.path.join(DATA_DIR, val_name), data[val_idx])
    np.save(os.path.join(DATA_DIR, test_name), data[test_idx])

    print(f"{source_file}: {len(data)} beats -> "
          f"val={len(val_idx)} ({val_name}), test={len(test_idx)} ({test_name})")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "test.npy")):
        raise FileNotFoundError(
            "data/test.npy not found. Run preprocess.py first."
        )

    split_and_save("test.npy", "val_normal.npy", "test_normal.npy")
    split_and_save("anomaly.npy", "val_anomaly.npy", "test_anomaly.npy")

    print("\nDone.")
    print("train.py now validates on data/val_normal.npy (early stopping only).")
    print("evaluate.py tunes the threshold on val_normal/val_anomaly, then")
    print("reports final metrics on test_normal/test_anomaly — a set that")
    print("never influenced training or threshold selection.")
