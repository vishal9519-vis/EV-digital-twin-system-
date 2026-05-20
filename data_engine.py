"""
data_engine.py
--------------
Generates, saves, and loads training datasets for the ML models.

Architecture separation:
  battery_simulator.py  →  physics / sensor model
  data_engine.py        →  dataset creation, feature engineering, persistence
  ml_models.py          →  model training and inference

Run directly to regenerate the dataset:
  python modules/data_engine.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from modules.battery_simulator import BatterySimulator
except ModuleNotFoundError:
    from battery_simulator import BatterySimulator

DATA_DIR     = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATASET_PATH = DATA_DIR / "battery_telemetry.csv"


def generate_dataset(n_vehicles=8, steps_per_vehicle=2000, profiles=None):
    """
    Simulate n_vehicles vehicles across different driving profiles and
    merge into one dataset. Multiple vehicles with different starting
    conditions make the ML models more robust.
    """
    if profiles is None:
        profiles = [
            "mixed", "city", "highway", "mixed",
            "city", "highway", "mixed", "charging"
        ]

    all_frames = []
    for vehicle_id, profile in enumerate(profiles[:n_vehicles]):
        print(f"  Simulating vehicle {vehicle_id + 1}/{n_vehicles} — profile: {profile}")

        sim = BatterySimulator(
            capacity_kwh   = np.random.uniform(60, 100),
            initial_soc    = np.random.uniform(0.40, 0.95),
            cycle_count    = np.random.randint(0, 400),
            ambient_temp_c = np.random.uniform(18, 42),
        )

        df                    = sim.run_session(n_steps=steps_per_vehicle, driving_profile=profile)
        df["vehicle_id"]      = vehicle_id
        df["driving_profile"] = profile
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  Dataset shape: {combined.shape}")
    return combined


def engineer_features(df):
    """
    Create derived features that improve ML model accuracy.

    Raw voltage and current are less informative than computed metrics.
    Feature engineering is often the biggest lever in ML projects.
    """
    df = df.copy()

    # Power in kW (positive = discharging, negative = charging)
    df["power_kw"] = (df["voltage_v"] * df["current_a"]) / 1000

    # Rolling stats over the last 10 readings to capture trends
    df["temp_rolling_mean"] = df["temperature_c"].rolling(10, min_periods=1).mean()
    df["temp_rolling_max"]  = df["temperature_c"].rolling(10, min_periods=1).max()

    # Rate of SOC change each step
    df["soc_delta"] = df["soc_pct"].diff().fillna(0)

    # How many degrees above the 35°C comfort ceiling
    df["thermal_stress"] = (df["temperature_c"] - 35).clip(lower=0)

    # Proxy for internal resistance growth — key ageing indicator
    df["voltage_sag"] = 400 - df["voltage_v"]

    # Binary flags
    df["is_fast_charge"] = (df["is_charging"] & (df["load_kw"] > 50)).astype(int)
    df["is_high_speed"]  = (df["speed_kmh"] > 100).astype(int)

    # Cumulative energy consumed since session start
    df["cumulative_kwh"] = df["load_kw"].clip(lower=0).cumsum() / 3600

    return df


def build_target_variables(df):
    """
    Add supervised-learning target columns.
    Each ML model needs a clear 'answer' to learn from.
    """
    df = df.copy()

    # Regression target: remaining range in km
    df["target_range_km"] = df["range_km"]

    # Regression target: battery temperature 60 seconds ahead
    df["target_temp_future"] = df["temperature_c"].shift(-60).fillna(df["temperature_c"])

    # Classification target: Good / Fair / Poor health bucket
    df["health_category"] = pd.cut(
        df["health_pct"],
        bins=[0, 70, 85, 100],
        labels=["Poor", "Fair", "Good"],
    )

    # Binary target: will the battery overheat?
    df["overheat_risk"] = (df["temperature_c"] > 42).astype(int)

    return df


def save_dataset(df, path=DATASET_PATH):
    df.to_csv(path, index=False)
    print(f"  Dataset saved → {path}  ({len(df):,} rows)")


def load_dataset(path=DATASET_PATH):
    if not path.exists():
        print("  No dataset found. Generating now ...")
        df = generate_dataset()
        df = engineer_features(df)
        df = build_target_variables(df)
        save_dataset(df)
        return df
    return pd.read_csv(path)


if __name__ == "__main__":
    print("=== EV Digital Twin — Data Generation ===\n")
    raw  = generate_dataset()
    feat = engineer_features(raw)
    full = build_target_variables(feat)
    save_dataset(full)
    print("\nSample:")
    print(full[["soc_pct", "voltage_v", "temperature_c", "range_km",
                "health_pct", "power_kw", "thermal_stress"]].head(5).to_string(index=False))
    print("\nDone.")
