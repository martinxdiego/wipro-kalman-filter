from __future__ import annotations
import numpy as np
import time
import matplotlib.pyplot as plt
from kalmed.data.simulate import HRSimConfig, simulate_hr
from kalmed.filters.ekf import EKF

# ======== Modellfunktionen ======== #
def f(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]]) @ x
def F_jac(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]])
def h(x): return (np.array([[1.0, 0.0]]) @ x).ravel()[0]
def H_jac(x): return np.array([[1.0, 0.0]])

# ======== Echtzeit-Demo ======== #
def main():
    # 10 Hz = 0.1 s Abtastzeit, 60 s Gesamtdauer
    cfg = HRSimConfig(n=600, dt=0.05, dropout_prob=0.05, spike_prob=0.02, r_meas=9.0)
    truth, meas, A, Q, H, r = simulate_hr(cfg)

    x0 = np.array([cfg.level0, cfg.trend0])
    P0 = np.diag([25.0, 1.0])
    ekf = EKF(x0, P0)
    R = np.array([[r]])

    # Plot-Setup
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live Herzfrequenz (bpm)")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Herzfrequenz")
    line_meas, = ax.plot([], [], "o-", color="orange", alpha=0.4, label="Messung")
    line_est,  = ax.plot([], [], "-",  color="green", lw=2, label="EKF-Schätzung")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(50, 150)

    t_vals, meas_vals, est_vals = [], [], []

    print("Starte Echtzeit-Simulation (10 Hz) …\n")
    for t in range(cfg.n):
        start = time.perf_counter()

        z = meas[t]
        ekf.predict(f, F_jac, Q)
        ekf.update(z, h, H_jac, R)
        est = ekf.state.x[0]

        t_now = t * cfg.dt
        t_vals.append(t_now)
        meas_vals.append(z)
        est_vals.append(est)

        # Plot aktualisieren
        line_meas.set_data(t_vals, meas_vals)
        line_est.set_data(t_vals, est_vals)
        if t_now > 10:
            ax.set_xlim(t_now - 10, t_now + 1)
        plt.pause(0.001)

        print(f"t={t_now:5.1f}s | Messung={z:6.2f} | Schätzung={est:6.2f}")

        # 10 Hz-Takt synchronisieren
        elapsed = time.perf_counter() - start
        time.sleep(max(0, cfg.dt - elapsed))

    print("\nFertig ✅ – 60 s Echtzeit-Plot beendet.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
