from __future__ import annotations
import numpy as np
import time
from kalmed.data.simulate import HRSimConfig, simulate_hr
from kalmed.filters.ekf import EKF


# lineares Modell (gleich wie vorher)
def f(x, u): return np.array([[1.0, 1.0],[0.0, 1.0]]) @ x
def F_jac(x, u): return np.array([[1.0, 1.0],[0.0, 1.0]])
def h(x): return (np.array([[1.0, 0.0]]) @ x).ravel()[0]
def H_jac(x): return np.array([[1.0, 0.0]])


def main():
    # vorbereitete „wahre“ Signale + Messungen
    cfg = HRSimConfig(n=60, dropout_prob=0.05, spike_prob=0.02, r_meas=9.0)
    truth, meas, A, Q, H, r = simulate_hr(cfg)

    x0 = np.array([cfg.level0, cfg.trend0])
    P0 = np.diag([25.0, 1.0])
    ekf = EKF(x0, P0)
    R = np.array([[r]])

    print("Starte Echtzeit-Simulation ...\n")
    for t in range(cfg.n):
        # ⏱️ Zeitmessung pro Iteration
        t_start = time.perf_counter()

        z = meas[t]
        ekf.predict(f, F_jac, Q)
        ekf.update(z, h, H_jac, R)

        est = ekf.state.x[0]
        print(f"t={t:02d}s | Messung={z:6.2f} | Schätzung={est:6.2f}")

        # simuliert Sensor-Takt: 1 s pro Messung
        t_elapsed = (time.perf_counter() - t_start)
        time.sleep(max(0, 1.0 - t_elapsed))  # bleibt synchron zur Echtzeit

    print("\nFertig ✅ – 60 s Echtzeit-Simulation abgeschlossen.")


if __name__ == "__main__":
    main()
