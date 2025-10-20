from __future__ import annotations
import numpy as np
from kalmed.data.simulate import simulate_hr, HRSimConfig
from kalmed.data.simulate_rr import simulate_rr, RRSimConfig
from kalmed.filters.ekf import EKF
from kalmed.eval.metrics import rmse
import matplotlib.pyplot as plt


# Modelle (gleich wie bisher)
def f(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]]) @ x
def F_jac(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]])
def h(x): return (np.array([[1.0, 0.0]]) @ x).ravel()[0]
def H_jac(x): return np.array([[1.0, 0.0]])


def main():
    # Simulationen
    hr_cfg = HRSimConfig(n=600, dt=0.5)
    rr_cfg = RRSimConfig(n=600, dt=0.5)

    hr_truth, hr_meas, A, Q_hr, H, r_hr = simulate_hr(hr_cfg)
    rr_truth, rr_meas, A, Q_rr, H, r_rr = simulate_rr(rr_cfg)

    # EKF für HR
    ekf_hr = EKF(np.array([hr_cfg.level0, hr_cfg.trend0]), np.diag([25.0, 1.0]))
    R_hr = np.array([[r_hr]])
    hr_est = np.zeros(hr_meas.shape)
    for t, z in enumerate(hr_meas):
        ekf_hr.predict(f, F_jac, Q_hr)
        ekf_hr.update(z, h, H_jac, R_hr)
        hr_est[t] = ekf_hr.state.x[0]

    # EKF für RR
    ekf_rr = EKF(np.array([rr_cfg.base_rate, 0.0]), np.diag([9.0, 0.5]))
    R_rr = np.array([[r_rr]])
    rr_est = np.zeros(rr_meas.shape)
    for t, z in enumerate(rr_meas):
        ekf_rr.predict(f, F_jac, Q_rr)
        ekf_rr.update(z, h, H_jac, R_rr)
        rr_est[t] = ekf_rr.state.x[0]

    # Ergebnisse anzeigen
    print(f"HR RMSE: {rmse(hr_truth[0, :], hr_est):.2f}")
    print(f"RR RMSE: {rmse(rr_truth[0, :], rr_est):.2f}")

    # Gemeinsamer Plot
    t = np.arange(len(hr_est)) * hr_cfg.dt
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_hr = "tab:red"
    ax1.set_xlabel("Zeit (s)")
    ax1.set_ylabel("Herzfrequenz [bpm]", color=color_hr)
    ax1.plot(t, hr_meas, ":", color=color_hr, alpha=0.4, label="HR Messung")
    ax1.plot(t, hr_est, "-", color=color_hr, lw=2, label="HR EKF")
    ax1.tick_params(axis="y", labelcolor=color_hr)

    ax2 = ax1.twinx()
    color_rr = "tab:blue"
    ax2.set_ylabel("Atemfrequenz [Atemzüge/min]", color=color_rr)
    ax2.plot(t, rr_meas, ":", color=color_rr, alpha=0.4, label="RR Messung")
    ax2.plot(t, rr_est, "-", color=color_rr, lw=2, label="RR EKF")
    ax2.tick_params(axis="y", labelcolor=color_rr)

    fig.tight_layout()
    plt.title("Herz- und Atemfrequenz – gefiltert mit EKF")
    plt.show()


if __name__ == "__main__":
    main()
