![CI](https://github.com/vishal9519-vis/EV-digital-twin-system-/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green) 
# ⚡ EV Digital Twin System

**AI-Powered Predictive Battery Intelligence & Smart Energy Simulation Platform**

A full-stack Python project that builds a virtual digital twin of an EV battery —
simulating, predicting, and visualising battery health, thermal behaviour, charging
efficiency, and long-term degradation in real time.

---

## What Is a Digital Twin?

A digital twin is a virtual replica of a physical system running in parallel with
the real thing. For an EV battery, the twin ingests sensor data (voltage, current,
temperature, SOC) and uses it to:

- **Monitor** current battery state
- **Predict** behaviour from 60 seconds to 12 months ahead
- **Optimise** charging strategy and driving behaviour
- **Alert** on risks before they become failures

The same concept is used in industry by Bosch, Tata Elxsi, KPIT, and major OEMs.

---

## Features

| Module | What It Does |
|---|---|
| Battery Simulator | Physics-based sensor data generator — mimics a real BMS |
| Data Engine | Dataset generation and feature engineering |
| ML Models | RandomForest range prediction + GradientBoosting thermal forecast |
| Thermal Intelligence | Risk scoring, cooling status, overheating prediction |
| Digital Twin Engine | Week-by-week health and range degradation forecast |
| Charging Intelligence | CC-CV curve simulation, urgency scoring, target SOC |
| Alert System | Deterministic rule-based risk alert engine |
| Live Dashboard | Streamlit dashboard with real-time Plotly charts |

---

## Project Structure

```
ev_digital_twin/
   │
   ├── app.py                       # Main Streamlit dashboard
   ├── setup.py                     # One-time data generation + model training
   ├── battery_simulator.py         # Physics simulation engine
   ├── data_engine.py               # Dataset generation and feature engineering
   ├── ml_models.py                 # ML model training and inference
   ├── thermal_intelligence.py      # Thermal risk analysis
   ├── alert_system.py              # Rule-based alert engine
   ├── charging_intelligence.py     # Charging strategy and CC-CV simulation
   ├── digital_twin.py              # Long-horizon degradation forecasting
   ├── requirements.txt
   ├── README.md
   └── .gitignore
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/vishal/ev-digital-twin.git
cd ev-digital-twin
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate data and train models (run once)

```bash
python setup.py
```

This takes about 30–60 seconds and creates:
- `data/battery_telemetry.csv` — 16,000 rows of synthetic telemetry
- `models/range_model.pkl` — RandomForest range predictor
- `models/thermal_model.pkl` — GradientBoosting thermal forecaster
- `models/health_model.pkl` — Health state classifier

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How to Use the Dashboard

1. **Sidebar** — adjust speed, A/C, charging mode, and charger power
2. **Step button** — advance simulation by one time step
3. **Auto-run toggle** — runs simulation continuously
4. Switch tabs to explore different modules
5. **Digital Twin tab** — change charging habit and climate for scenario comparison

---

## Tech Stack

| Technology | Why |
|---|---|
| Python 3.10+ | Core language |
| NumPy / Pandas | Numerical simulation and data handling |
| Scikit-learn | ML model training (RandomForest, GradientBoosting) |
| Streamlit | Interactive web dashboard — no JavaScript needed |
| Plotly | Interactive charts (gauge, line, pie, dual-axis) |
| Joblib | Trained model serialisation |

---

## ML Models

| Model | Algorithm | Target | Metric |
|---|---|---|---|
| Range Predictor | RandomForest (120 trees) | km remaining | MAE ~2-5 km |
| Thermal Forecaster | GradientBoosting (100 trees) | Temp in 60s | MAE ~0.5-1.5°C |
| Health Classifier | RandomForest (100 trees) | Good/Fair/Poor | Accuracy ~99% |

---

## Interview Talking Points

**Why rule-based alerts instead of ML?**
Safety-critical thresholds must be deterministic and auditable. An ML model could
silently miss a 48°C temperature event if that combination was rare in training.
Hard rules guarantee limits are always enforced.

**Why dedicated models instead of one multi-output model?**
Each target has a different distribution and error metric. Dedicated models with
tuned hyperparameters outperform a single generic model.

**What makes this a digital twin vs a dashboard?**
A dashboard shows current state. A digital twin simulates future state — you can
ask what happens under different charging habits and get a year-long projection
in seconds rather than waiting for real data.

**How would you scale this to real vehicles?**
Replace `BatterySimulator.step()` with a CAN-bus reader or MQTT subscriber.
Everything downstream — feature engineering, ML inference, alerts — works unchanged.

---

## Future Improvements

- Connect to real CAN-bus data using python-can library
- Add federated learning to train across a fleet without sharing raw data
- Build a REST API with FastAPI for mobile app integration
- Integrate MQTT for live streaming from physical sensors
- Implement Kalman Filter for more accurate SOC estimation
- Add Docker support for fleet-scale cloud deployment

---

## License

MIT License — free for personal and commercial use.
