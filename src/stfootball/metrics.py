import numpy as np


def ade(pred: np.ndarray, true: np.ndarray, nodes=slice(0, 22)) -> float:
    """Average displacement error in metres over all future steps (players only by default)."""
    return float(np.linalg.norm(pred[:, :, nodes] - true[:, :, nodes], axis=-1).mean())


def fde(pred: np.ndarray, true: np.ndarray, nodes=slice(0, 22)) -> float:
    """Final displacement error in metres at the last future step."""
    return float(np.linalg.norm(pred[:, -1, nodes] - true[:, -1, nodes], axis=-1).mean())
