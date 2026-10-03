from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # pitch & sampling
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    raw_fps: int = 25
    target_fps: int = 10
    # cleaning
    max_speed: float = 12.0  # m/s; faster jumps are treated as tracking glitches
    smooth_window_s: float = 0.5
    # windows (RQ1: 4s observed -> 2s predicted)
    input_s: float = 4.0
    output_s: float = 2.0
    stride_s: float = 1.0


DEFAULT = Config()
