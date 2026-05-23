"""
thermal_intelligence.py
-----------------------
Thermal risk analysis, scoring, and cooling recommendations.

Battery temperature zones:
  < 10°C   : Cold — lithium plating risk during charging, low capacity
  10–35°C  : Optimal operating window
  35–42°C  : Elevated — degradation accelerates
  42–48°C  : Warning — reduce load immediately
  > 48°C   : Critical — thermal runaway risk
"""

from dataclasses import dataclass


@dataclass
class ThermalReport:
    temperature_c:    float
    predicted_temp_c: float
    risk_score:       float
    alert_level:      str
    efficiency_score: float
    recommendations:  list
    cooling_status:   str


def analyse_thermal_state(
    current_temp,
    predicted_temp,
    ambient_temp,
    speed_kmh,
    is_charging,
    charging_power_kw=0,
):
    """
    Compute a full thermal report from current sensor values.

    Risk score combines:
      - How close current temp is to the 45°C danger threshold
      - Rate of temperature rise (predicted - current)
      - Fast-charging penalty (biggest thermal stress source)
      - Ambient temperature penalty (less cooling headroom when hot outside)
    """

    # Normalise temperature to 0-100 danger scale (25°C = 0, 45°C = 100)
    temp_danger    = max(0.0, min(100.0, (current_temp - 25) / 20 * 100))

    # Rate of rise penalty — how fast is it heating up?
    rate_of_rise   = max(0.0, predicted_temp - current_temp)
    rise_penalty   = min(30.0, rate_of_rise * 10)

    # Fast-charging generates the most heat
    if is_charging and charging_power_kw > 50:
        charge_penalty = min(25.0, (charging_power_kw - 50) / 4)
    else:
        charge_penalty = 0.0

    # Hot ambient means the cooling system has less headroom
    ambient_penalty = max(0.0, (ambient_temp - 30) * 0.5)

    risk_score = round(min(100.0, temp_danger + rise_penalty + charge_penalty + ambient_penalty), 1)

    if risk_score < 25:
        alert_level = "Safe"
    elif risk_score < 50:
        alert_level = "Watch"
    elif risk_score < 75:
        alert_level = "Warning"
    else:
        alert_level = "Critical"

    # Thermal efficiency: higher speed = more airflow = better cooling
    base_efficiency = 85.0
    if current_temp > 40:
        base_efficiency -= (current_temp - 40) * 3
    if speed_kmh > 60:
        base_efficiency += (speed_kmh - 60) * 0.05
    efficiency_score = round(max(40.0, min(100.0, base_efficiency)), 1)

    recs = _build_recommendations(
        current_temp, predicted_temp, alert_level,
        is_charging, charging_power_kw, ambient_temp, speed_kmh,
    )

    if current_temp < 30:
        cooling_status = "Passive / Idle"
    elif current_temp < 38:
        cooling_status = "Active Cooling — Low"
    elif current_temp < 43:
        cooling_status = "Active Cooling — High"
    else:
        cooling_status = "Maximum Cooling — EMERGENCY"

    return ThermalReport(
        temperature_c    = round(current_temp, 2),
        predicted_temp_c = round(predicted_temp, 2),
        risk_score       = risk_score,
        alert_level      = alert_level,
        efficiency_score = efficiency_score,
        recommendations  = recs,
        cooling_status   = cooling_status,
    )


def _build_recommendations(
    current_temp, predicted_temp, alert_level,
    is_charging, charging_power_kw, ambient_temp, speed_kmh,
):
    recs = []

    if alert_level == "Safe":
        recs.append("Thermal system operating within normal parameters.")
        recs.append("No action required — continue driving.")

    if alert_level in ("Watch", "Warning"):
        recs.append("Battery temperature is elevated. Monitor closely.")
        if speed_kmh < 40:
            recs.append("Increase speed slightly to improve airflow cooling.")
        if is_charging and charging_power_kw > 80:
            recs.append("Consider switching to a lower-power charger (50 kW or below).")
        if ambient_temp > 35:
            recs.append("Park in shade or a covered bay to reduce ambient heat load.")

    if alert_level == "Critical":
        recs.append("CRITICAL: Battery temperature dangerously high. Take immediate action.")
        recs.append("Stop fast charging immediately — switch to slow AC charging or stop.")
        recs.append("Reduce vehicle speed to lower motor load and heat generation.")
        recs.append("Pull over safely and allow the pack to cool for 15-20 minutes.")
        recs.append("Contact service if temperature does not drop within 5 minutes.")

    if predicted_temp > current_temp + 3:
        recs.append(
            f"Temperature projected to reach {predicted_temp:.1f} C within 60 seconds."
        )

    if current_temp < 10:
        recs.append("Battery is cold. Range and charging speed are reduced until pack warms up.")
        recs.append("Enable battery pre-conditioning before starting a fast-charge session.")

    return recs


def thermal_history_summary(temps):
    """Summarise a list of temperature readings for the analytics panel."""
    if not temps:
        return {}

    import numpy as np
    arr = np.array(temps, dtype=float)

    return {
        "mean_temp_c":   round(float(arr.mean()), 2),
        "max_temp_c":    round(float(arr.max()), 2),
        "min_temp_c":    round(float(arr.min()), 2),
        "pct_above_40c": round(float((arr > 40).mean() * 100), 1),
        "pct_above_45c": round(float((arr > 45).mean() * 100), 1),
    }
