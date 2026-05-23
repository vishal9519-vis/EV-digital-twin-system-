import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.battery_simulator import BatterySimulator


def test_soc_decreases_when_driving():
    sim = BatterySimulator(initial_soc=0.80)
    before = sim.soc
    sim.step(speed_kmh=80, ac_on=False, is_charging=False)
    assert sim.soc < before


def test_soc_increases_when_charging():
    sim = BatterySimulator(initial_soc=0.50)
    before = sim.soc
    sim.step(speed_kmh=0, ac_on=False, is_charging=True, charging_power_kw=50)
    assert sim.soc > before


def test_soc_never_exceeds_1():
    sim = BatterySimulator(initial_soc=0.99)
    for _ in range(100):
        sim.step(speed_kmh=0, is_charging=True, charging_power_kw=150)
    assert sim.soc <= 1.0


def test_voltage_in_reasonable_range():
    sim = BatterySimulator(initial_soc=0.80)
    snap = sim.step(speed_kmh=60)
    assert 330 <= snap["voltage_v"] <= 420


def test_health_degrades_with_more_cycles():
    sim_new = BatterySimulator(cycle_count=0)
    sim_old = BatterySimulator(cycle_count=500)
    assert sim_old.health < sim_new.health


def test_get_snapshot_returns_dict():
    sim = BatterySimulator()
    snap = sim.get_snapshot()
    assert isinstance(snap, dict)
    assert "soc_pct" in snap
    assert "voltage_v" in snap
    assert "temperature_c" in snap
