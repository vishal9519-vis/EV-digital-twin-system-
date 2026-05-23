"""
charging_intelligence.py
------------------------
Smart charging recommendations and CC-CV charge curve simulation.

Key concepts:
  CC-CV charging : Constant Current until ~80% SOC, then taper to
                   Constant Voltage to avoid overcharging
  Optimal window : 20-80% SOC preserves cell chemistry long-term
  De-rating      : Real BMS systems reduce charge power at extreme temps
                   and near 100% SOC to protect the cells
"""

from dataclasses import dataclass


@dataclass
class ChargingReport:
    urgency_score:                float
    urgency_label:                str
    recommended_target_soc:       float
    estimated_time_to_full_min:   float
    estimated_time_to_target_min: float
    charge_efficiency_pct:        float
    recommendations:              list
    optimal_charge_window:        tuple


def analyse_charging_state(
    soc_pct,
    battery_capacity_kwh,
    health_pct,
    temperature_c,
    charger_power_kw,
    next_trip_km=100.0,
    energy_consumption_kwh_per_km=0.18,
):
    """
    Produce a full charging intelligence report.

    Urgency is driven by how much energy you need for the next trip
    versus how much you have right now, plus raw SOC thresholds.
    """

    usable_capacity  = battery_capacity_kwh * (health_pct / 100)
    energy_needed    = next_trip_km * energy_consumption_kwh_per_km
    energy_available = (soc_pct / 100) * usable_capacity
    trip_coverage    = energy_available / max(energy_needed, 0.1)

    # Urgency score
    if soc_pct < 10:
        urgency_score = 100.0
    elif soc_pct < 20:
        urgency_score = 85.0
    elif trip_coverage < 1.1:
        urgency_score = 70.0
    elif soc_pct < 40:
        urgency_score = 45.0
    elif soc_pct < 60:
        urgency_score = 20.0
    else:
        urgency_score = max(0.0, 10.0 - (soc_pct - 60) * 0.2)

    if urgency_score > 75:
        urgency_label = "CHARGE NOW"
    elif urgency_score > 45:
        urgency_label = "Charge Soon"
    elif urgency_score > 20:
        urgency_label = "Plan a Charge"
    else:
        urgency_label = "No Urgency"

    # Target SOC: 80% for daily longevity, more if trip demands it
    energy_with_buffer = energy_needed * 1.15
    required_soc       = (energy_with_buffer / usable_capacity) * 100
    target_soc         = round(min(95.0, max(80.0, required_soc)), 1)

    effective_power = _effective_charge_power(charger_power_kw, temperature_c, soc_pct)

    energy_to_full   = (1 - soc_pct / 100) * usable_capacity
    energy_to_target = max(0.0, (target_soc - soc_pct) / 100 * usable_capacity)

    time_to_full_min   = (energy_to_full   / max(effective_power, 0.1)) * 60
    time_to_target_min = (energy_to_target / max(effective_power, 0.1)) * 60

    # Round-trip charging efficiency drops at temperature extremes
    base_eff = 96.0
    if temperature_c < 15:
        base_eff -= (15 - temperature_c) * 0.4
    elif temperature_c > 40:
        base_eff -= (temperature_c - 40) * 0.5
    charge_efficiency = round(max(80.0, base_eff), 1)

    recs = _build_charge_recommendations(
        soc_pct, urgency_score, temperature_c,
        charger_power_kw, effective_power,
        target_soc, time_to_target_min, health_pct,
    )

    return ChargingReport(
        urgency_score                = round(urgency_score, 1),
        urgency_label                = urgency_label,
        recommended_target_soc       = target_soc,
        estimated_time_to_full_min   = round(time_to_full_min, 0),
        estimated_time_to_target_min = round(time_to_target_min, 0),
        charge_efficiency_pct        = charge_efficiency,
        recommendations              = recs,
        optimal_charge_window        = (20.0, 80.0),
    )


def _effective_charge_power(power_kw, temp_c, soc_pct):
    """
    De-rate charging power for temperature and SOC.

    Real BMS systems implement detailed de-rating tables. This is a
    simplified version capturing the two most important effects:
      - Cold or hot battery → slower safe charging
      - CV phase taper above 80% SOC → power tapers to ~30% at 100%
    """
    p = power_kw

    if temp_c < 10:
        p *= 0.3
    elif temp_c < 20:
        p *= 0.6 + (temp_c - 10) * 0.04
    elif temp_c > 45:
        p *= 0.4
    elif temp_c > 38:
        p *= 1 - (temp_c - 38) * 0.03

    if soc_pct > 80:
        taper = 1 - (soc_pct - 80) / 20 * 0.7
        p *= max(0.30, taper)

    return max(0.5, p)


def _build_charge_recommendations(
    soc, urgency, temp, charger_kw, effective_kw,
    target_soc, time_min, health,
):
    recs = []

    if urgency > 75:
        recs.append("Plug in immediately — battery is too low for safe driving.")
    elif urgency > 45:
        recs.append("Connect to a charger at your next opportunity.")
    else:
        recs.append("Battery level is adequate. Charge at your convenience.")

    recs.append(
        f"Charging to {target_soc:.0f}% recommended. "
        "Keeping daily charges below 80% significantly extends battery life."
    )

    if temp < 10:
        recs.append(
            "Battery is cold. Pre-condition the pack for 15 minutes "
            "before DC fast charging to avoid lithium plating."
        )
    elif temp > 40:
        recs.append(
            "Battery is warm. Rest the vehicle for 10 minutes "
            "before charging to improve efficiency."
        )

    if effective_kw < charger_kw * 0.7:
        recs.append(
            f"Charger de-rated from {charger_kw:.0f} kW to "
            f"~{effective_kw:.0f} kW due to thermal or SOC limits. This is normal."
        )

    recs.append(f"Estimated time to reach target SOC: {int(time_min)} minutes.")

    if health < 80:
        recs.append(
            "Degraded battery detected. Use slow AC charging where possible "
            "to reduce further stress on aged cells."
        )

    recs.append("Ideal daily charging window: keep battery between 20% and 80%.")

    return recs


def simulate_charge_curve(
    start_soc,
    capacity_kwh,
    charger_power_kw,
    temperature_c,
    target_soc=100.0,
    step_minutes=1.0,
):
    """
    Simulate a CC-CV charging session minute by minute.
    Returns a list of dicts for dashboard plotting.
    """
    soc   = start_soc
    curve = []
    t     = 0.0

    while soc < target_soc and t < 600:
        effective = _effective_charge_power(charger_power_kw, temperature_c, soc)
        delta_kwh = effective * (step_minutes / 60)
        soc      += (delta_kwh / capacity_kwh) * 100
        soc       = min(soc, 100.0)

        curve.append({
            "time_min":  round(t, 1),
            "soc_pct":   round(soc, 2),
            "power_kw":  round(effective, 1),
        })
        t += step_minutes

    return curve
