"""
digital_twin.py
---------------
Long-horizon battery degradation forecasting — the core of the
'Digital Twin' concept.

A regular dashboard shows current state.
A digital twin simulates FUTURE state:
  - Battery health 6 months from now
  - Expected range after 1 year of use
  - Degradation trajectory under different charging habits

The model is empirical (based on published ageing studies) rather
than full electrochemical simulation. Real OEM digital twins at
Bosch, Mahindra, and Tata Elxsi use similar hybrid approaches.

Reference: Birkl et al. (2017), "Degradation diagnostics for lithium
ion cells", Journal of Power Sources.
"""

from dataclasses import dataclass


@dataclass
class TwinForecast:
    weeks:               list
    soh_timeline:        list
    range_timeline:      list
    efficiency_timeline: list
    cycle_timeline:      list
    weeks_to_80pct_soh:  float
    weeks_to_70pct_soh:  float
    final_soh:           float
    final_range_km:      float


def run_future_simulation(
    current_soh,
    current_range_km,
    current_cycles,
    battery_capacity_kwh,
    charging_habit="mixed",
    climate="tropical",
    weekly_km=500.0,
    weeks=52,
):
    """
    Project battery health and range degradation week by week.

    Two main ageing mechanisms modelled:
      1. Cycle ageing    — each charge cycle costs ~0.03% capacity
      2. Calendar ageing — time degrades cells even when parked

    Both are scaled by:
      - charging_habit : how well the user manages charge levels
      - climate        : ambient temperature (biggest external factor)
    """

    habit_multipliers = {
        "longevity":  0.60,   # charges to 80%, avoids DC fast charging
        "mixed":      1.00,   # typical user
        "aggressive": 1.65,   # frequent fast charges, often charges to 100%
    }

    climate_multipliers = {
        "cold":     0.85,   # cold slows calendar ageing
        "temperate":1.00,   # baseline
        "tropical": 1.30,   # hot and humid accelerates ageing
        "hot_arid": 1.45,   # worst case — very hot and dry
    }

    habit_mult   = habit_multipliers.get(charging_habit, 1.0)
    climate_mult = climate_multipliers.get(climate, 1.0)

    base_cycle_loss    = 0.03 * habit_mult * climate_mult   # % per cycle
    base_calendar_loss = 0.05 * climate_mult                # % per week

    # Estimate cycles per week from driving distance and pack size
    consumption_km  = 6.0   # approx km per kWh
    cycles_per_week = weekly_km / (battery_capacity_kwh * consumption_km)

    # Nominal efficiency for range calculation
    nominal_eff = (battery_capacity_kwh / max(current_range_km, 1)) * 100

    soh               = current_soh
    cumulative_cycles = current_cycles
    range_km          = current_range_km

    soh_timeline        = []
    range_timeline      = []
    efficiency_timeline = []
    cycle_timeline      = []
    week_list           = list(range(0, weeks + 1))

    weeks_to_80 = None
    weeks_to_70 = None

    for w in week_list:
        soh_timeline.append(round(soh, 2))
        range_timeline.append(round(range_km, 1))

        # Efficiency degrades as health declines (more resistance, less usable energy)
        eff_factor = 1 + (1 - soh / 100) * 0.4
        efficiency_timeline.append(round(nominal_eff * eff_factor, 2))
        cycle_timeline.append(int(cumulative_cycles))

        if soh <= 80.0 and weeks_to_80 is None:
            weeks_to_80 = float(w)
        if soh <= 70.0 and weeks_to_70 is None:
            weeks_to_70 = float(w)

        # Advance one week
        cycle_loss    = cycles_per_week * base_cycle_loss
        calendar_loss = base_calendar_loss

        # Non-linear ageing cliff: degradation accelerates below 75% SoH
        if soh < 75:
            cycle_loss    *= 1.5
            calendar_loss *= 1.3

        soh               = max(50.0, soh - cycle_loss - calendar_loss)
        cumulative_cycles += cycles_per_week
        range_km           = current_range_km * (soh / current_soh)

    return TwinForecast(
        weeks               = week_list,
        soh_timeline        = soh_timeline,
        range_timeline      = range_timeline,
        efficiency_timeline = efficiency_timeline,
        cycle_timeline      = cycle_timeline,
        weeks_to_80pct_soh  = weeks_to_80 if weeks_to_80 is not None else float(weeks),
        weeks_to_70pct_soh  = weeks_to_70 if weeks_to_70 is not None else float(weeks),
        final_soh           = soh_timeline[-1],
        final_range_km      = range_timeline[-1],
    )


def compare_scenarios(current_soh, current_range_km, current_cycles, capacity_kwh, weeks=52):
    """
    Run all three charging habits and return a dict for comparison charts.
    """
    return {
        habit: run_future_simulation(
            current_soh, current_range_km, current_cycles,
            capacity_kwh, charging_habit=habit, weeks=weeks,
        )
        for habit in ("longevity", "mixed", "aggressive")
    }


def degradation_summary_text(forecast, habit):
    """
    Generate a plain-language AI insight paragraph for the dashboard.
    """
    lines = []
    w80   = forecast.weeks_to_80pct_soh

    if w80 < 52:
        lines.append(
            f"With your current {habit} charging habits, the battery is projected "
            f"to drop below 80% health in approximately {int(w80)} weeks "
            f"({int(w80 / 4.3)} months)."
        )
    else:
        lines.append(
            f"With {habit} charging, battery health is expected to stay "
            "above 80% for over a year."
        )

    lines.append(
        f"Estimated range at end of forecast: {forecast.final_range_km:.0f} km "
        f"(currently {forecast.range_timeline[0]:.0f} km)."
    )

    if habit == "aggressive":
        lines.append(
            "Switching to a longevity charging profile could extend "
            "battery service life by 30-50%."
        )
    elif habit == "longevity":
        lines.append(
            "Excellent charging behaviour. Staying between 20-80% SOC "
            "and avoiding frequent fast charging maximises lifespan."
        )

    return "  ".join(lines)
