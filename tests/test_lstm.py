"""LSTM baseline tests on synthetic windows (skipped without the `ml` extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stfootball.config import Config  # noqa: E402
from stfootball.models.lstm import PlayerLSTM  # noqa: E402
from stfootball.train import TrainConfig, fit, predict, split_train_val  # noqa: E402
from stfootball.windows import WindowSet  # noqa: E402

CFG = Config()
T_IN, T_OUT = int(CFG.input_s * CFG.target_fps), int(CFG.output_s * CFG.target_fps)


def linear_windows(n=40, seed=0, match_id="syn") -> WindowSet:
    """Every node moves in a straight line with a random constant velocity."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-40, 40, (n, 1, 23, 2))
    vel = rng.uniform(-3, 3, (n, 1, 23, 2))
    t = (np.arange(T_IN + T_OUT) / CFG.target_fps)[None, :, None, None]
    pos = start + vel * t
    X = np.concatenate([pos[:, :T_IN], np.broadcast_to(vel, (n, T_IN, 23, 2))], -1)
    return WindowSet(X, pos[:, T_IN:], np.zeros((n, 23), int), np.arange(n), match_id)


def test_forward_shape_and_zero_offset_identity():
    model = PlayerLSTM(T_OUT, hidden=16, layers=1)
    X = torch.as_tensor(linear_windows(4).X[:, :, :22], dtype=torch.float32)
    out = model(X)
    assert out.shape == (4, T_OUT, 22, 2)
    torch.nn.init.zeros_(model.head[-1].weight); torch.nn.init.zeros_(model.head[-1].bias)
    assert torch.allclose(model(X), X[:, -1:, :, :2].expand(-1, T_OUT, -1, -1))


def test_split_keeps_gap_between_train_and_val():
    w = linear_windows(100)
    (Xtr, _), (Xva, _) = split_train_val([w], 0.1, CFG)
    assert len(Xva) == 10 and len(Xtr) == 100 - 10 - 5


def test_fit_learns_linear_motion():
    sets = [linear_windows(200, seed=s, match_id=f"m{s}") for s in range(2)]
    tc = TrainConfig(epochs=8, batch_size=32, lr=3e-3, patience=8, device="cpu")
    model = fit(PlayerLSTM(T_OUT, hidden=32, layers=1), sets, tc, log=lambda *_: None)
    test = linear_windows(50, seed=9)
    before = np.linalg.norm(test.X[:, -1:, :22, :2] - test.Y[:, :, :22], axis=-1).mean()
    after = np.linalg.norm(predict(model, test.X, device="cpu") - test.Y[:, :, :22], axis=-1).mean()
    assert after < 0.5 * before, (before, after)
