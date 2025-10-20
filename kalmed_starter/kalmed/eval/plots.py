import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_outdir(d="outputs"):
    os.makedirs(d, exist_ok=True)
    return d

def plot_timeseries(truth, noisy, ekf, outdir="outputs", fname="timeseries.png"):
    out = ensure_outdir(outdir)
    t = np.arange(len(truth))
    plt.figure()
    plt.plot(t, truth, label="Ground Truth")
    plt.plot(t, noisy, label="Noisy Measurement", alpha=0.5)
    plt.plot(t, ekf, label="EKF Estimate")
    plt.xlabel("t (samples)"); plt.ylabel("HR (bpm)"); plt.legend(); plt.tight_layout()
    path = os.path.join(out, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    return path
