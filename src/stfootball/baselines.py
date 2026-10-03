"""Non-learned baselines for RQ1."""
import numpy as np


def constant_velocity(X: np.ndarray, t_out: int, fps: float) -> np.ndarray:
    """Extrapolate the last observed position with the last observed velocity."""
    last = X[:, -1]                      # (N, 23, 4)
    steps = np.arange(1, t_out + 1)[None, :, None, None] / fps
    return last[:, None, :, :2] + last[:, None, :, 2:] * steps


def stationary(X: np.ndarray, t_out: int) -> np.ndarray:
    return np.repeat(X[:, -1:, :, :2], t_out, axis=1)
