from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Tuple

DATASET_PATH = Path.cwd() / "cache" / "dataset.npz"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_sample(label: str, landmarks: np.ndarray) -> None:
    X, y = load_dataset()
    if X.size == 0:
        X = landmarks.reshape(1, -1)
        y = np.array([label])
    else:
        X = np.vstack([X, landmarks.reshape(1, -1)])
        y = np.concatenate([y, np.array([label])])
    np.savez_compressed(DATASET_PATH, X=X, y=y)


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    if DATASET_PATH.exists():
        data = np.load(DATASET_PATH, allow_pickle=False)
        return data["X"], data["y"]
    else:
        return np.empty((0, 63), dtype=np.float32), np.array([], dtype=str)
