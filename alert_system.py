"""
alert_system.py
---------------
Deterministic rule-based safety alert engine.

Why rule-based instead of ML?
  Safety-critical thresholds must always fire reliably. An ML model
  trained on typical data could silently miss a 48°C temperature spike
  if that combination of features was rare in training. Hard rules
  guarantee hard limits are never missed.

Each alert carries:
  severity  : INFO / WARNING / CRITICAL
  code      : unique identifier (useful for logging and filtering)
  message   : what happened
  action    : what the driver should do
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Alert:
    severity:  str
    code:      str
    message:   str
    action:    str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self):
        return {
            "severity":  self.severity,
            "code":      self.code,
            "message":   self.message,
            "action":    self.action,
            "timestamp": self.timestamp,
        }


class AlertEngine:
    """
    Evaluates a sensor snapshot against the rule table.
    Returns alerts sorted by severity (CRITICAL first).
    """

    def evaluate(self, snapshot):
        alerts    = []
        soc       = snapshot.get("soc_pct", 100)
        temp      = snapshot.get("temperature_c", 25)
        voltage   = snapshot.get("voltage_v", 400)
        health    = snapshot.get("health_pct", 100)
        range_km  = snapshot.get("range_km", 999)
        charging  = snapshot.get("is_charging", False)
        speed     = snapshot.get("speed_kmh", 0)
        current   = snapshot.get("current_a", 0)

        # ----- SOC / Range ------------------------------------------------
        if soc <= 5:
            alerts.append(Alert(
                "CRITICAL", "SOC_CRITICAL",
                f"Battery critically low: {soc:.1f}%",
                "Find the nearest charger immediately. Do not continue driving.",
            ))
        elif soc <= 15:
            alerts.append(Alert(
                "WARNING", "SOC_LOW",
                f"Battery low: {soc:.1f}%",
                "Begin navigation to a charging station within the next 10-15 km.",
            ))
        elif soc <= 25:
            alerts.append(Alert(
                "INFO", "SOC_WATCH",
                f"Battery below 25%: {soc:.1f}%",
                "Plan a charging stop soon.",
            ))

        if range_km < 20 and not charging:
            alerts.append(Alert(
                "CRITICAL", "LOW_RANGE",
                f"Range critically low: {range_km:.0f} km",
                "Stop and charge immediately.",
            ))
        elif range_km < 50 and not charging:
            alerts.append(Alert(
                "WARNING", "RANGE_WARNING",
                f"Range below 50 km: {range_km:.0f} km",
                "Locate a charging station and reduce speed to extend range.",
            ))

        # ----- Thermal ----------------------------------------------------
        if temp > 48:
            alerts.append(Alert(
                "CRITICAL", "OVERHEAT_CRITICAL",
                f"Battery dangerously hot: {temp:.1f} C",
                "Stop the vehicle safely. Stop charging. Call roadside assistance.",
            ))
        elif temp > 42:
            alerts.append(Alert(
                "WARNING", "OVERHEAT_WARNING",
                f"Battery temperature high: {temp:.1f} C",
                "Reduce speed. Stop fast charging. Allow passive cooling.",
            ))
        elif temp > 38:
            alerts.append(Alert(
                "INFO", "TEMP_WATCH",
                f"Temperature approaching limit: {temp:.1f} C",
                "Monitor and avoid hard acceleration.",
            ))

        if temp < 5 and charging:
            alerts.append(Alert(
                "WARNING", "COLD_CHARGE",
                f"Charging in cold conditions: {temp:.1f} C",
                "Enable battery pre-heating before DC fast charging.",
            ))

        # ----- Voltage ----------------------------------------------------
        if voltage < 340 and not charging:
            alerts.append(Alert(
                "WARNING", "LOW_VOLTAGE",
                f"Pack voltage low: {voltage:.1f} V",
                "Charge immediately to avoid deep discharge cell damage.",
            ))

        # ----- Health / Degradation ---------------------------------------
        if health < 70:
            alerts.append(Alert(
                "CRITICAL", "HEALTH_CRITICAL",
                f"Battery health severely degraded: {health:.1f}%",
                "Schedule battery inspection and possible replacement.",
            ))
        elif health < 80:
            alerts.append(Alert(
                "WARNING", "HEALTH_WARNING",
                f"Battery health degraded: {health:.1f}%",
                "Avoid fast charging. Use eco mode to extend remaining life.",
            ))

        # ----- Driving behaviour ------------------------------------------
        if speed > 130 and not charging:
            alerts.append(Alert(
                "INFO", "HIGH_SPEED",
                f"High speed: {speed:.0f} km/h. Range drops significantly above 100 km/h.",
                "Reduce speed below 100 km/h to improve energy efficiency.",
            ))

        if abs(current) > 250:
            alerts.append(Alert(
                "WARNING", "HIGH_CURRENT",
                f"High current: {abs(current):.0f} A",
                "Reduce acceleration or charging power to protect cell chemistry.",
            ))

        # Sort: CRITICAL → WARNING → INFO
        order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        alerts.sort(key=lambda a: order[a.severity])
        return alerts
