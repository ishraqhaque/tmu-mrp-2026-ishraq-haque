"""Chronological train/test split and expanding-window walk-forward folds.

Time-series evaluation must never shuffle: every test point must lie strictly
after its training data. ``walk_forward_folds`` yields expanding-window folds used
for robustness and as the paired samples for the significance test (each fold gives
one hybrid-vs-baseline comparison on the same out-of-sample block).
"""

from __future__ import annotations

import numpy as np


def chronological_split(n: int, train_frac: float = 0.70) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positional indices for a single 70/30-style split."""
    n_train = int(n * train_frac)
    return np.arange(n_train), np.arange(n_train, n)


def walk_forward_folds(
    n: int,
    n_folds: int = 5,
    train_frac: float = 0.70,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over the held-out tail.

    The first ``train_frac`` of the data is the initial training window; the
    remaining tail is divided into ``n_folds`` contiguous test blocks. Fold *i*
    trains on everything before block *i* and tests on block *i*.
    """
    n_train0 = int(n * train_frac)
    test_idx_all = np.arange(n_train0, n)
    blocks = np.array_split(test_idx_all, n_folds)
    folds = []
    for block in blocks:
        if len(block) == 0:
            continue
        train_idx = np.arange(0, block[0])      # expanding window up to the block
        folds.append((train_idx, block))
    return folds
