"""
ml_models.py
------------
Machine learning layer for the EV Digital Twin.

Three dedicated models:
  1. RangePredictor     — RandomForest regression     → km remaining
  2. ThermalPredictor   — GradientBoosting regression → temperature 60s ahead
  3. HealthClassifier   — RandomForest classification → Good / Fair / Poor

Why separate models instead of one?
  Each target has a different distribution and loss metric. Dedicated
  models with tuned hyperparameters outperform a single generic model.

Why RandomForest and GradientBoosting?
  - Handle non-linear battery physics well
  - Robust to sensor noise (outliers)
  - Fast enough to retrain during a live demo
  - Feature importances are easy to explain in interviews
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

try:
    from modules.data_engine import load_dataset, engineer_features
except ModuleNotFoundError:
    from data_engine import load_dataset, engineer_features

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature lists for each model
RANGE_FEATURES = [
    "soc_pct", "voltage_v", "temperature_c", "speed_kmh",
    "health_pct", "load_kw", "power_kw", "soc_delta",
    "thermal_stress", "is_high_speed", "ac_on",
]

THERMAL_FEATURES = [
    "temperature_c", "current_a", "speed_kmh", "load_kw",
    "soc_pct", "ambient_temp_c", "thermal_stress",
    "temp_rolling_mean", "is_fast_charge",
]

HEALTH_FEATURES = [
    "cycle_count", "temp_rolling_max", "voltage_sag",
    "thermal_stress", "soc_pct", "cumulative_kwh", "is_fast_charge",
]


def _safe_cols(df, wanted):
    """Return only columns that actually exist in the dataframe."""
    return [c for c in wanted if c in df.columns]


def train_range_model(df):
    """
    RandomForest regression for remaining range prediction.
    Bagging (training many trees on random subsets) reduces variance
    and prevents overfitting to one driving pattern.
    """
    features = _safe_cols(df, RANGE_FEATURES)
    X = df[features].fillna(0)
    y = df["target_range_km"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,     # number of trees
        max_depth=12,         # cap tree depth to prevent overfitting
        min_samples_leaf=4,   # each leaf needs at least 4 samples
        random_state=42,
        n_jobs=-1,            # use all CPU cores
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  [RangePredictor]   MAE={mae:.1f} km   R²={r2:.3f}")

    joblib.dump((model, features), MODEL_DIR / "range_model.pkl")
    return model, features, mae, r2


def train_thermal_model(df):
    """
    GradientBoosting for short-horizon temperature forecasting.
    Sequential tree building corrects residuals from previous trees —
    often beats RandomForest on tabular regression.
    """
    features = _safe_cols(df, THERMAL_FEATURES)
    X = df[features].fillna(0)
    y = df["target_temp_future"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.08,   # smaller = more careful, less overfitting
        max_depth=5,
        subsample=0.8,        # use 80% of data per tree (stochastic GB)
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  [ThermalPredictor] MAE={mae:.2f} °C  R²={r2:.3f}")

    joblib.dump((model, features), MODEL_DIR / "thermal_model.pkl")
    return model, features, mae, r2


def train_health_classifier(df):
    """
    RandomForest classification: Good / Fair / Poor battery health.
    Outputs a class label and per-class probabilities.
    """
    features = _safe_cols(df, HEALTH_FEATURES)
    X = df[features].fillna(0)
    y = df["health_category"].astype(str)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  [HealthClassifier] Accuracy={acc:.3f}")

    joblib.dump((model, features, le), MODEL_DIR / "health_model.pkl")
    return model, features, le, acc


def train_all_models(force=False):
    """
    Train all three models. Skips if saved files exist unless force=True.
    Returns a dict of evaluation metrics.
    """
    models_exist = all([
        (MODEL_DIR / "range_model.pkl").exists(),
        (MODEL_DIR / "thermal_model.pkl").exists(),
        (MODEL_DIR / "health_model.pkl").exists(),
    ])

    if models_exist and not force:
        print("  Models already exist. Pass force=True to retrain.")
        return {}

    print("=== Training ML Models ===\n")
    df = load_dataset()

    # Ensure engineered features are present
    for col in ["power_kw", "thermal_stress", "voltage_sag"]:
        if col not in df.columns:
            df = engineer_features(df)
            break

    # Convert boolean columns to int for sklearn
    for col in ["ac_on", "is_charging"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    _, _, mae_r, r2_r = train_range_model(df)
    _, _, mae_t, r2_t = train_thermal_model(df)
    _, _, _, acc_h    = train_health_classifier(df)

    metrics = {
        "range_mae_km":    round(mae_r, 2),
        "range_r2":        round(r2_r, 3),
        "thermal_mae_c":   round(mae_t, 3),
        "thermal_r2":      round(r2_t, 3),
        "health_accuracy": round(acc_h, 3),
    }

    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    train_all_models(force=True)
