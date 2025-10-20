from __future__ import annotations
import numpy as np
import time
import matplotlib
# Automatische Backend-Erkennung (funktioniert auf allen Systemen)
for backend in ["TkAgg", "Qt5Agg", "WXAgg"]:
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt

from kalmed.data.simulate import simulate_hr, HRSimConfig
from kalmed.data.simulate_rr import simulate_rr, RRSimConfig
from kalmed.filters.ekf import EKF

# ======== Modellfunktionen ======== #
def f(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]]) @ x
def F_jac(x, u): return np.array([[1.0, 1.0], [0.0, 1.0]])
def h(x): return (np.array([[1.0, 0.0]]) @ x).ravel()[0]
def H_jac(x): return np.array([[1.0, 0.0]])

# ======== Echtzeit-Demo ======== #
def main():
    # 10 Hz Takt
    hr_cfg = HRSimConfig(n=600, dt=0.1, dropout_prob=0.05, spike_prob=0.02)
    rr_cfg = RRSimConfig(n=600, dt=0.1, dropout_prob=0.03, spike_prob=0.01)

    hr_truth, hr_meas, A, Q_hr, H, r_hr = simulate_hr(hr_cfg)
    rr_truth, rr_meas, A, Q_rr, H, r_rr = simulate_rr(rr_cfg)

    ekf_hr = EKF(np.array([hr_cfg.level0, hr_cfg.trend0]), np.diag([25.0, 1.0]))
    ekf_rr = EKF(np.array([rr_cfg.base_rate, 0.0]), np.diag([9.0, 0.5]))
    R_hr = np.array([[r_hr]])
    R_rr = np.array([[r_rr]])

    # Plot vorbereiten
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    ax.set_title("Live-Überwachung: Herz- & Atemfrequenz")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Herzfrequenz (bpm)", color="tab:red")
    ax2.set_ylabel("Atemfrequenz (Atemzüge/min)", color="tab:blue")

    line_hr, = ax.plot([], [], "-", color="tab:red", lw=2, label="HR EKF")
    line_rr, = ax2.plot([], [], "-", color="tab:blue", lw=2, label="RR EKF")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    t_vals, hr_vals, rr_vals = [], [], []

    # Erzwinge erstes Zeichnen (damit Fenster nicht leer bleibt)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print("Starte Echtzeit-Monitor (10 Hz) …\n")

    for t in range(hr_cfg.n):
        start = time.perf_counter()

        # HR und RR filtern
        ekf_hr.predict(f, F_jac, Q_hr)
        ekf_hr.update(hr_meas[t], h, H_jac, R_hr)
        hr_val = ekf_hr.state.x[0]

        ekf_rr.predict(f, F_jac, Q_rr)
        ekf_rr.update(rr_meas[t], h, H_jac, R_rr)
        rr_val = ekf_rr.state.x[0]

        # Daten anhängen
        t_now = t * hr_cfg.dt
        t_vals.append(t_now)
        hr_vals.append(hr_val)
        rr_vals.append(rr_val)

        # Linien aktualisieren
        line_hr.set_data(t_vals, hr_vals)
        line_rr.set_data(t_vals, rr_vals)

        # Achsenbereich
        if t_now > 10:
            ax.set_xlim(t_now - 10, t_now + 1)
            ax2.set_xlim(t_now - 10, t_now + 1)
        else:
            ax.set_xlim(0, 10)
            ax2.set_xlim(0, 10)

        ax.set_ylim(50, 150)
        ax2.set_ylim(10, 50)

        # Forciertes Redraw
        plt.draw()
        plt.gcf().canvas.start_event_loop(0.001)

        # Konsolenausgabe
        print(f"t={t_now:5.1f}s | HR={hr_val:6.2f} bpm | RR={rr_val:6.2f} /min")

        # Echtzeit-Takt
        elapsed = time.perf_counter() - start
        time.sleep(max(0, hr_cfg.dt - elapsed))

    print("\nFertig ✅ – Echtzeit-Monitor beendet.")
    plt.ioff()
    plt.show(block=True)

if __name__ == "__main__":
    main()
