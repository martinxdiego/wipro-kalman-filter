from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class HRSimConfig:
    n: int = 2000              # number of samples
    dt: float = 1.0            # sampling interval (s)
    level0: float = 100.0      # initial HR (bpm)
    trend0: float = 0.0        # initial bpm/s
    q_level: float = 0.02      # process noise (level)
    q_trend: float = 0.001     # process noise (trend)
    r_meas: float = 4.0        # measurement noise (variance, bpm^2)
    dropout_prob: float = 0.05 # probability of missing measurement
    spike_prob: float = 0.01   # probability of spike artifact
    spike_scale: float = 25.0  # magnitude of spikes
    clip_min: float = 50.0     # physiological reasonable bounds
    clip_max: float = 200.0

def simulate_hr(cfg: HRSimConfig, rng: np.random.Generator | None = None):
    """Simulate latent HR [level, trend] and noisy measurements with artefacts."""
    rng = np.random.default_rng() if rng is None else rng
    n, dt = cfg.n, cfg.dt

    # State: x = [level, trend]^T with local-trend model
    A = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=float)
    Q = np.array([[cfg.q_level, 0.0],
                  [0.0, cfg.q_trend]], dtype=float)

    H = np.array([[1.0, 0.0]], dtype=float)  # measure level only

    x = np.zeros((2, n))
    z = np.full(n, np.nan)  # measurements (with dropouts)
    x[:,0] = [cfg.level0, cfg.trend0]

    # process
    for t in range(1, n):
        w = rng.multivariate_normal(mean=[0.0,0.0], cov=Q)
        x[:,t] = A @ x[:,t-1] + w

        # physiologic soft clamp in latent to avoid runaway
        x[0,t] = np.clip(x[0,t], cfg.clip_min, cfg.clip_max)

    # measurements with Gaussian noise
    for t in range(n):
        if rng.random() < cfg.dropout_prob:
            z[t] = np.nan  # missing
            continue
        y = (H @ x[:,t]).item() + rng.normal(0.0, np.sqrt(cfg.r_meas))
        # add spikes
        if rng.random() < cfg.spike_prob:
            y += rng.normal(0.0, cfg.spike_scale)
        # sensor clipping
        y = float(np.clip(y, cfg.clip_min, cfg.clip_max))
        z[t] = y

    return x, z, A, Q, H, cfg.r_meas
