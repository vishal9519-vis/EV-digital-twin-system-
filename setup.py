"""
setup.py
--------
One-time setup. Run this BEFORE launching the dashboard.

  python setup.py

This will:
  1. Simulate 8 virtual vehicles and generate training data
  2. Train 3 ML models and save them to disk

Then launch the dashboard with:
  streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  EV Digital Twin System — Setup & Model Training")
print("=" * 55)

print("\n[1/2] Generating telemetry dataset ...")
from modules.data_engine import (
    generate_dataset,
    engineer_features,
    build_target_variables,
    save_dataset,
)

raw  = generate_dataset(n_vehicles=8, steps_per_vehicle=2000)
feat = engineer_features(raw)
full = build_target_variables(feat)
save_dataset(full)

print("\n[2/2] Training ML models ...")
from modules.ml_models import train_all_models
metrics = train_all_models(force=True)

print("\n" + "=" * 55)
print("  Setup complete!")
print("  Launch the dashboard with:")
print("    streamlit run app.py")
print("=" * 55)
