from __future__ import annotations
import numpy as np
import time
from kalmed.data.simulate import simulate_hr, HRSimConfig
from kalmed.filters.ekf import EKF
from kalmed.eval.metrics import rmse
from kalmed.eval.plots import plot_timeseries


def f(x, u):
    # linear local-trend with dt=1 encoded in F below; here we do it explicit for clarity
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    return F @ x

def F_jac(x, u):
    return np.array([[1.0, 1.0],
                     [0.0, 1.0]])

def h(x):
    H = np.array([[1.0, 0.0]])
    return (H @ x).ravel()[0]

def H_jac(x):
    return np.array([[1.0, 0.0]])


def main():
    # 🕒 Start-Zeit
    start = time.perf_counter()

    cfg = HRSimConfig(n=1500, dropout_prob=0.08, spike_prob=0.02, r_meas=9.0)
    truth, meas, A, Q, H, r = simulate_hr(cfg)

    # EKF init
    x0 = np.array([cfg.level0, cfg.trend0])
    P0 = np.diag([25.0, 1.0])
    ekf = EKF(x0, P0)
    R = np.array([[r]])

    est = np.zeros(meas.shape)
    for t, z in enumerate(meas):
        ekf.predict(f, F_jac, Q)
        ekf.update(z, h, H_jac, R)
        est[t] = ekf.state.x[0]

    # Metriken
    mask = ~np.isnan(meas)
    r_truth = truth[0, :]
    print("RMSE (EKF vs truth):", rmse(r_truth, est))

    # Plot
    out = plot_timeseries(r_truth, np.where(mask, meas, np.nan), est)
    print("Plot gespeichert:", out)

    # ⏱️ End-Zeit
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    print(f"Berechnungszeit: {total_ms:.2f} ms  ({total_ms / cfg.n:.4f} ms/Sample)")


if __name__ == "__main__":
    main()
