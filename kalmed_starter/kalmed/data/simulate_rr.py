from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class RRSimConfig:
    n: int = 600              # number of samples (~10 min)
    dt: float = 1.0           # sampling interval (s)
    base_rate: float = 25.0   # breaths per minute
    amplitude: float = 2.5    # natural variability (bpm)
    q_level: float = 0.002
    q_trend: float = 0.0001
    r_meas: float = 1.0
    dropout_prob: float = 0.03
    spike_prob: float = 0.005
    spike_scale: float = 5.0
    clip_min: float = 10.0
    clip_max: float = 60.0

def simulate_rr(cfg: RRSimConfig, rng: np.random.Generator | None = None):
    """Simulate breathing-rate signal (RR) with noise, dropouts & artefacts."""
    rng = np.random.default_rng() if rng is None else rng
    n, dt = cfg.n, cfg.dt

    # Latent true respiration pattern: sinusoidal variation around base_rate
    t = np.arange(n) * dt
    base = cfg.base_rate + cfg.amplitude * np.sin(2*np.pi*t/40)  # 40 s rhythm

    x = np.zeros((2, n))
    z = np.full(n, np.nan)
    x[:, 0] = [base[0], 0.0]

    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    Q = np.array([[cfg.q_level, 0.0],
                  [0.0, cfg.q_trend]])
    H = np.array([[1.0, 0.0]])

    for t in range(1, n):
        w = rng.multivariate_normal(mean=[0.0, 0.0], cov=Q)
        x[:, t] = A @ x[:, t-1] + w
        x[0, t] += base[t] - base[t-1]  # drive towards sinus baseline
        x[0, t] = np.clip(x[0, t], cfg.clip_min, cfg.clip_max)

    for t in range(n):
        if rng.random() < cfg.dropout_prob:
            z[t] = np.nan
            continue
        y = (H @ x[:, t]).item() + rng.normal(0.0, np.sqrt(cfg.r_meas))
        if rng.random() < cfg.spike_prob:
            y += rng.normal(0.0, cfg.spike_scale)
        y = float(np.clip(y, cfg.clip_min, cfg.clip_max))
        z[t] = y

    return x, z, A, Q, H, cfg.r_meas
