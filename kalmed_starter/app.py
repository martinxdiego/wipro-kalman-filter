import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import json
import shutil
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from kalmed.data.simulate import simulate_hr, HRSimConfig
from kalmed.data.simulate_rr import simulate_rr, RRSimConfig
from kalmed.filters.ekf import EKF
from kalmed.eval.metrics import rmse

# ======================================================
# 🧠 MODEL-FUNKTIONEN (SYSTEMGLEICHUNGEN)
# ======================================================
def f_model(x, u):
    return np.array([[1.0, 1.0], [0.0, 1.0]]) @ x

def F_jac(x, u):
    return np.array([[1.0, 1.0], [0.0, 1.0]])

def h_meas(x):
    return (np.array([[1.0, 0.0]]) @ x).ravel()[0]

def H_jac(x):
    return np.array([[1.0, 0.0]])

# ======================================================
# ⚙️ APP CONFIG
# ======================================================
st.set_page_config(page_title="HSLU WIPRO Kalman Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ======================================================
# 🎓 HEADER – HSLU WIPRO BRANDING (Base64-Logo)
# ======================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
logo_b64 = get_base64_image(logo_path)

st.markdown(f"""
    <style>
    .header-container {{
        width: 100%;
        background: linear-gradient(90deg, #ffffff 0%, #f0f7f7 60%, #00b4b6 100%);
        padding: 1.2rem 2rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .header-left {{
        display: flex;
        align-items: center;
    }}
    .header-left img {{
        height: 55px;
        margin-right: 1rem;
    }}
    .header-title {{
        color: #202124;
        font-size: 1.6rem;
        font-weight: 600;
    }}
    .header-subtitle {{
        color: #5f6368;
        font-size: 0.9rem;
        margin-top: -4px;
    }}
    .header-right {{
        color: #202124;
        font-size: 0.85rem;
        opacity: 0.8;
        text-align: right;
    }}
    </style>

    <div class="header-container">
        <div class="header-left">
            <img src="data:image/png;base64,{logo_b64}" alt="HSLU Logo">
            <div>
                <div class="header-title">HSLU – WIPRO Kalman Filter Dashboard</div>
                <div class="header-subtitle">Echtzeitanalyse von Herz- und Atemfrequenz mit Extended Kalman Filter</div>
            </div>
        </div>
        <div class="header-right">
            Hochschule Luzern – Informatik<br>Wirtschaftsprojekt HS25
        </div>
    </div>
""", unsafe_allow_html=True)

# ======================================================
# 🌞 HELLERES MEDICAL-THEME
# ======================================================
st.markdown("""
    <style>
    body { background-color: #f8f9fa; color: #202124; font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] { background-color: #f8f9fa; color: #202124; }
    [data-testid="stHeader"] { background: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f1f3f4; color: #202124; }
    .stMarkdown, .stText, .stDataFrame, .stTable, .stPlotlyChart { color: #202124 !important; }
    div.block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { background-color: #00b4b6; color: white; border-radius: 6px; border: none; font-weight: 600; transition: 0.2s; }
    .stButton>button:hover { background-color: #009497; transform: scale(1.02); }
    h1, h2, h3 { color: #00b4b6; font-weight: 600; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p { color: #202124; }
    .plot-container { background-color: #ffffff !important; border-radius: 12px; padding: 10px; border: 1px solid #ddd; }
    .stProgress > div > div > div { background-color: #00b4b6; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 📂 BASISORDNER
# ======================================================
output_dir = os.path.join(os.getcwd(), "outputs")
profiles_dir = os.path.join(os.getcwd(), "profiles")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(profiles_dir, exist_ok=True)

# ======================================================
# 📊 TABS
# ======================================================
tab_sim, tab_prof, tab_reports = st.tabs(["📊 Simulation", "💾 Profile", "📄 Reports"])

# ======================================================
# 🧩 TAB 1: SIMULATION
# ======================================================
with tab_sim:
    st.title("📊 Echtzeit-Simulation – Kalman-Filter Herz & Atemfrequenz")

    # --- Erklärung für Nutzer:innen ---
    st.markdown("""
    ### 🔍 Interpretation der Analyse
    Im Folgenden werden zwei Signale in Echtzeit dargestellt:

    - **Linke Grafik (Herzfrequenz)** zeigt, wie der Kalman-Filter das verrauschte Herzfrequenzsignal (*bpm = beats per minute*) glättet.
    - **Rechte Grafik (Atemfrequenz)** zeigt dasselbe Prinzip für die Atemrate (*breaths/min*).

    **Achsenbedeutung:**
    - **x-Achse:** Zeit in Sekunden (Verlauf der Simulation)
    - **y-Achse:** Frequenzwerte in physiologischen Einheiten  
      (Herzfrequenz = Schläge/Minute, Atemfrequenz = Atemzüge/Minute)

    Die blauen Linien repräsentieren das *gefilterte Signal* (Kalman-Schätzung),  
    während die helleren Linien das *Rohsignal mit Messrauschen* zeigen.
    """)

    # Sidebar Parameter
      # Sidebar Parameter mit Erklärungen
    st.sidebar.header("⚙️ Simulationseinstellungen")

    dt = st.sidebar.slider(
        "Abtastzeit (dt in s)",
        0.05, 0.5, 0.1, 0.05,
        help="Definiert, wie oft pro Sekunde Messwerte simuliert werden. "
             "Ein kleinerer Wert bedeutet höhere zeitliche Auflösung, aber auch mehr Rechenaufwand."
    )

    r_meas = st.sidebar.slider(
        "Sensorrauschen R",
        1.0, 15.0, 6.0, 0.5,
        help="Stellt die Stärke des Messrauschens dar. "
             "Ein höherer Wert simuliert ungenauere Sensoren, wodurch der Kalman-Filter stärker glätten muss."
    )

    dropout_prob = st.sidebar.slider(
        "Dropout-Wahrscheinlichkeit",
        0.0, 0.2, 0.05, 0.01,
        help="Simuliert Datenlücken (z. B. durch Signalverlust). "
             "Gibt an, wie oft Messwerte zufällig ausfallen."
    )

    spike_prob = st.sidebar.slider(
        "Spike-Wahrscheinlichkeit",
        0.0, 0.1, 0.02, 0.005,
        help="Bestimmt, wie oft fehlerhafte Ausreißerwerte (‚Spikes‘) im Signal auftreten. "
             "Ideal zum Testen der Robustheit des Filters."
    )

    n = st.sidebar.slider(
        "Anzahl Samples (Dauer)",
        100, 1000, 600, 50,
        help="Anzahl der Messpunkte, die simuliert werden. "
             "Gibt indirekt auch die Simulationsdauer an (Abtastzeit × Anzahl Samples)."
    )

    flush_interval = st.sidebar.slider(
        "Speicher-Intervall (s)",
        2, 20, 10, 1,
        help="Bestimmt, in welchem Zeitintervall Zwischenergebnisse auf die Festplatte geschrieben werden. "
             "Ein kleinerer Wert erhöht die Datensicherheit, aber auch die Schreibhäufigkeit."
    )


    col_btn1, col_btn2 = st.columns([1, 1])
    start = col_btn1.button("▶️ Start Simulation")
    stop = col_btn2.button("⏹️ Stop")

    col1, col2 = st.columns(2)
    chart_hr, chart_rr = col1.empty(), col2.empty()
    status, metrics = st.empty(), st.empty()

    if start and not stop:
        st.session_state.running = True
        status.info("Simulation läuft – Echtzeit-Vergleich aktiv.")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        temp_path = os.path.join(output_dir, f"live_{timestamp}_temp.csv")
        live_path = temp_path.replace("_temp.csv", ".csv")

        with open(temp_path, "w") as fout:
            fout.write("Zeit (s),HR roh,HR gefiltert,RR roh,RR gefiltert\n")

        hr_cfg = HRSimConfig(n=n, dt=dt, r_meas=r_meas,
                             dropout_prob=dropout_prob, spike_prob=spike_prob)
        rr_cfg = RRSimConfig(n=n, dt=dt, r_meas=r_meas/5,
                             dropout_prob=dropout_prob/2, spike_prob=spike_prob/2)

        hr_truth, hr_meas, _, Q_hr, _, r_hr = simulate_hr(hr_cfg)
        rr_truth, rr_meas, _, Q_rr, _, r_rr = simulate_rr(rr_cfg)
        ekf_hr = EKF(np.array([hr_cfg.level0, hr_cfg.trend0]), np.diag([25.0, 1.0]))
        ekf_rr = EKF(np.array([rr_cfg.base_rate, 0.0]), np.diag([9.0, 0.5]))
        R_hr, R_rr = np.array([[r_hr]]), np.array([[r_rr]])

        df_hr = pd.DataFrame(columns=["t", "HR [bpm] roh", "HR [bpm] gefiltert"])
        df_rr = pd.DataFrame(columns=["t", "RR [breaths/min] roh", "RR [breaths/min] gefiltert"])
        chart_hr_obj = chart_hr.line_chart(df_hr[["HR [bpm] roh", "HR [bpm] gefiltert"]])
        chart_rr_obj = chart_rr.line_chart(df_rr[["RR [breaths/min] roh", "RR [breaths/min] gefiltert"]])

        hr_est, rr_est, buffer = [], [], []
        last_flush = time.time()

        for t in range(n):
            if stop or not st.session_state.get("running", True):
                status.warning("Simulation gestoppt.")
                break

            ekf_hr.predict(f_model, F_jac, Q_hr)
            ekf_hr.update(hr_meas[t], h_meas, H_jac, R_hr)
            hr_val = ekf_hr.state.x[0]

            ekf_rr.predict(f_model, F_jac, Q_rr)
            ekf_rr.update(rr_meas[t], h_meas, H_jac, R_rr)
            rr_val = ekf_rr.state.x[0]

            hr_est.append(hr_val)
            rr_est.append(rr_val)

            df_hr.loc[len(df_hr)] = [t*dt, hr_meas[t], hr_val]
            df_rr.loc[len(df_rr)] = [t*dt, rr_meas[t], rr_val]
            chart_hr_obj.add_rows(df_hr.tail(1)[["HR [bpm] roh", "HR [bpm] gefiltert"]])
            chart_rr_obj.add_rows(df_rr.tail(1)[["RR [breaths/min] roh", "RR [breaths/min] gefiltert"]])

            buffer.append(f"{t*dt:.3f},{hr_meas[t]:.3f},{hr_val:.3f},{rr_meas[t]:.3f},{rr_val:.3f}\n")

            if time.time() - last_flush >= flush_interval:
                with open(temp_path, "a") as fout:
                    fout.writelines(buffer)
                buffer.clear()
                last_flush = time.time()

            time.sleep(dt)

        if buffer:
            with open(temp_path, "a") as fout:
                fout.writelines(buffer)
        shutil.move(temp_path, live_path)

        mask_hr, mask_rr = ~np.isnan(hr_meas), ~np.isnan(rr_meas)
        rmse_hr = rmse(hr_truth[0, mask_hr], np.array(hr_est)[mask_hr])
        rmse_rr = rmse(rr_truth[0, mask_rr], np.array(rr_est)[mask_rr])
        raw_error_hr = rmse(hr_truth[0, mask_hr], hr_meas[mask_hr])
        raw_error_rr = rmse(rr_truth[0, mask_rr], rr_meas[mask_rr])
        improv_hr = 100*(1-rmse_hr/raw_error_hr)
        improv_rr = 100*(1-rmse_rr/raw_error_rr)

        metrics.success(
            f"**Filter-Güte:**\n- ❤️ HR RMSE: {rmse_hr:.2f} (→ {improv_hr:.1f}% besser)\n"
            f"- 💨 RR RMSE: {rmse_rr:.2f} (→ {improv_rr:.1f}% besser)"
        )
        st.success(f"✅ Simulation abgeschlossen. Live-Datei: `{live_path}`")

# ======================================================
# 💾 PROFILE TAB
# ======================================================
with tab_prof:
    st.title("💾 Parameter-Profile speichern & laden")
    profile_files = [f for f in os.listdir(profiles_dir) if f.endswith(".json")]
    selected_profile = st.selectbox("Profil auswählen:", ["– keines –"] + profile_files)
    col1, col2 = st.columns(2)
    load_profile = col1.button("📂 Laden")
    save_profile = col2.button("💾 Neues Profil speichern")

    if load_profile and selected_profile != "– keines –":
        with open(os.path.join(profiles_dir, selected_profile)) as fin:
            params = json.load(fin)
        st.json(params)
        st.success(f"✅ Profil '{selected_profile}' geladen!")

    if save_profile:
        profile_name = st.text_input("Profilname eingeben:")
        if profile_name:
            params = {"dt": dt, "r_meas": r_meas, "dropout_prob": dropout_prob,
                      "spike_prob": spike_prob, "n": n, "flush_interval": flush_interval}
            with open(os.path.join(profiles_dir, f"{profile_name}.json"), "w") as fout:
                json.dump(params, fout, indent=4)
            st.success(f"💾 Profil '{profile_name}.json' gespeichert!")

# ======================================================
# 📄 REPORTS TAB
# ======================================================
with tab_reports:
    st.title("📄 Generierte Reports & Exporte")
    reports = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]
    if reports:
        for file in sorted(reports, reverse=True):
            st.markdown(f"📎 **{file}** — [Download]({os.path.join(output_dir, file)})")
    else:
        st.info("Noch keine Reports vorhanden – führe zuerst eine Simulation aus.")

