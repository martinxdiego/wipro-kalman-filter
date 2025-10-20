from __future__ import annotations
import numpy as np
from kalmed.data.simulate_rr import simulate_rr, RRSimConfig
from kalmed.filters.ekf import EKF
from kalmed.eval.metrics import rmse
from kalmed.eval.plots import plot_timeseries

def f(x, u): return np.array([[1.0, 1.0],[0.0, 1.0]]) @ x
def F_jac(x, u): return np.array([[1.0, 1.0],[0.0, 1.0]])
def h(x): return (np.array([[1.0, 0.0]]) @ x).ravel()[0]
def H_jac(x): return np.array([[1.0, 0.0]])

def main():
    cfg = RRSimConfig()
    truth, meas, A, Q, H, r = simulate_rr(cfg)

    x0 = np.array([cfg.base_rate, 0.0])
    P0 = np.diag([9.0, 0.5])
    ekf = EKF(x0, P0)
    R = np.array([[r]])

    est = np.zeros(meas.shape)
    for t, z in enumerate(meas):
        ekf.predict(f, F_jac, Q)
        ekf.update(z, h, H_jac, R)
        est[t] = ekf.state.x[0]

    mask = ~np.isnan(meas)
    r_truth = truth[0, :]
    print("RMSE (EKF vs truth):", rmse(r_truth, est))
    out = plot_timeseries(r_truth, np.where(mask, meas, np.nan), est,
                          outdir="outputs", fname="rr_timeseries.png")
    print("Plot gespeichert:", out)

if __name__ == "__main__":
    main()
