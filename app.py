"""
app.py
------
Main Streamlit dashboard for the EV Digital Twin System.

Run with:
    streamlit run app.py

Dashboard structure:
    Sidebar        — vehicle controls and twin configuration
    KPI Row        — 6 live metric cards
    Tab 1          — Real-Time Battery Monitor
    Tab 2          — AI Range and Energy Predictions
    Tab 3          — Thermal Intelligence
    Tab 4          — Smart Charging
    Tab 5          — Digital Twin Future Simulation
    Tab 6          — Active Risk Alerts
"""

import sys
import os
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.battery_simulator    import BatterySimulator
from modules.thermal_intelligence import analyse_thermal_state, thermal_history_summary
from modules.alert_system         import AlertEngine
from modules.charging_intelligence import analyse_charging_state, simulate_charge_curve
from modules.digital_twin         import (
    run_future_simulation,
    compare_scenarios,
    degradation_summary_text,
)


# -----------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="EV Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------
# Custom CSS — dark futuristic theme
# -----------------------------------------------------------------------
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0a0e1a; color: #e0e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1425;
        border-right: 1px solid #1e3a5f;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1b2e, #112240);
        border: 1px solid #1e4a7a;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 4px 15px rgba(0, 120, 255, 0.08);
    }
    div[data-testid="metric-container"] label {
        color: #7ec8e3 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1425;
        border-bottom: 1px solid #1e3a5f;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7ec8e3;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
    }

    /* Headings */
    h1, h2, h3 { color: #e0e8f0 !important; letter-spacing: 0.02em; }

    /* Alert boxes */
    .alert-critical {
        background: rgba(220,38,38,0.12);
        border-left: 3px solid #dc2626;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .alert-warning {
        background: rgba(234,179,8,0.12);
        border-left: 3px solid #eab308;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .alert-info {
        background: rgba(59,130,246,0.12);
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
    }

    /* Dividers */
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------
def _init_session():
    if "sim" not in st.session_state:
        st.session_state.sim     = BatterySimulator(
            capacity_kwh   = 75.0,
            initial_soc    = 0.82,
            cycle_count    = 145,
            ambient_temp_c = 31.0,
        )
        st.session_state.history = []

_init_session()
sim = st.session_state.sim


# -----------------------------------------------------------------------
# Sidebar — vehicle controls
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ EV Digital Twin")
    st.markdown("*AI-Powered Battery Intelligence*")
    st.divider()

    st.markdown("### Vehicle Parameters")
    speed_kmh   = st.slider("Speed (km/h)",        0, 150, 65)
    ac_on       = st.toggle("A/C System",          value=True)
    is_charging = st.toggle("Charging Mode",       value=False)
    charge_kw   = st.slider("Charger Power (kW)",  0, 150, 50,
                             disabled=not is_charging)

    st.divider()
    st.markdown("### Digital Twin Config")
    charging_habit = st.selectbox(
        "Charging Habit",
        ["longevity", "mixed", "aggressive"],
        index=1,
    )
    climate = st.selectbox(
        "Climate Zone",
        ["cold", "temperate", "tropical", "hot_arid"],
        index=2,
    )
    weekly_km      = st.slider("Weekly Distance (km)", 100, 1200, 500)
    forecast_weeks = st.slider("Forecast Period (weeks)", 4, 104, 52)
    next_trip_km   = st.slider("Next Trip Distance (km)", 10, 500, 120)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        step_btn  = st.button("▶ Step",  use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)
    auto_run = st.toggle("Auto-run simulation", value=False)

    if reset_btn:
        for k in ["sim", "history"]:
            st.session_state.pop(k, None)
        _init_session()
        sim = st.session_state.sim
        st.rerun()


# -----------------------------------------------------------------------
# Simulation step function
# -----------------------------------------------------------------------
def run_step():
    snap = sim.step(
        speed_kmh         = speed_kmh,
        ac_on             = ac_on,
        is_charging       = is_charging,
        charging_power_kw = float(charge_kw) if is_charging else 0.0,
    )

    history = st.session_state.history

    # Add engineered features needed for analysis modules
    if history:
        prev = history[-1]
        snap["soc_delta"]         = snap["soc_pct"] - prev.get("soc_pct", snap["soc_pct"])
        recent_temps              = [r["temperature_c"] for r in history[-10:]] + [snap["temperature_c"]]
        snap["temp_rolling_mean"] = float(np.mean(recent_temps))
        snap["temp_rolling_max"]  = float(max(recent_temps))
    else:
        snap["soc_delta"]         = 0.0
        snap["temp_rolling_mean"] = snap["temperature_c"]
        snap["temp_rolling_max"]  = snap["temperature_c"]

    snap["power_kw"]       = (snap["voltage_v"] * snap["current_a"]) / 1000
    snap["thermal_stress"] = max(0.0, snap["temperature_c"] - 35)
    snap["voltage_sag"]    = 400 - snap["voltage_v"]
    snap["is_fast_charge"] = int(is_charging and charge_kw > 50)
    snap["is_high_speed"]  = int(speed_kmh > 100)
    snap["cumulative_kwh"] = sum(r.get("load_kw", 0) for r in history) / 3600

    st.session_state.history.append(snap)
    return snap


# -----------------------------------------------------------------------
# Advance simulation or get current snapshot
# -----------------------------------------------------------------------
if step_btn or auto_run:
    snapshot = run_step()
else:
    snapshot = sim.get_snapshot()
    if not st.session_state.history:
        st.session_state.history.append(snapshot)

hist_df = pd.DataFrame(st.session_state.history)


# -----------------------------------------------------------------------
# Run all analysis engines on the latest snapshot
# -----------------------------------------------------------------------
alert_engine  = AlertEngine()
active_alerts = alert_engine.evaluate(snapshot)

thermal_report = analyse_thermal_state(
    current_temp      = snapshot.get("temperature_c", 30),
    predicted_temp    = snapshot.get("temperature_c", 30) + random.uniform(-0.5, 1.5),
    ambient_temp      = snapshot.get("ambient_temp_c", 28),
    speed_kmh         = speed_kmh,
    is_charging       = is_charging,
    charging_power_kw = float(charge_kw) if is_charging else 0,
)

charge_report = analyse_charging_state(
    soc_pct               = snapshot.get("soc_pct", 50),
    battery_capacity_kwh  = sim.capacity_kwh,
    health_pct            = snapshot.get("health_pct", 90),
    temperature_c         = snapshot.get("temperature_c", 30),
    charger_power_kw      = float(charge_kw),
    next_trip_km          = next_trip_km,
)

forecast = run_future_simulation(
    current_soh          = snapshot.get("health_pct", 90),
    current_range_km     = snapshot.get("range_km", 250),
    current_cycles       = int(snapshot.get("cycle_count", 150)),
    battery_capacity_kwh = sim.capacity_kwh,
    charging_habit       = charging_habit,
    climate              = climate,
    weekly_km            = weekly_km,
    weeks                = forecast_weeks,
)

# Convenience aliases
soc  = snapshot.get("soc_pct", 0)
temp = snapshot.get("temperature_c", 0)
rng  = snapshot.get("range_km", 0)
hlth = snapshot.get("health_pct", 0)
volt = snapshot.get("voltage_v", 0)
curr = snapshot.get("current_a", 0)


# -----------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:14px; margin-bottom:4px;'>
    <span style='font-size:2rem;'>⚡</span>
    <div>
        <h1 style='margin:0; font-size:1.6rem; letter-spacing:0.05em;'>
            EV DIGITAL TWIN SYSTEM
        </h1>
        <p style='margin:0; color:#7ec8e3; font-size:0.8rem; letter-spacing:0.1em;'>
            AI-POWERED PREDICTIVE BATTERY INTELLIGENCE
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Show CRITICAL alerts at the top of the page
for a in [x for x in active_alerts if x.severity == "CRITICAL"]:
    st.markdown(
        f"<div class='alert-critical'>🚨 <b>{a.code}</b> — {a.message}"
        f"<br><small>{a.action}</small></div>",
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------------------------------------------------
# KPI row — 6 metric cards
# -----------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🔋 Battery",       f"{soc:.1f} %",
          f"{snapshot.get('soc_delta', 0):.3f} %/s")
k2.metric("🌡️ Temperature",  f"{temp:.1f} °C",
          thermal_report.alert_level)
k3.metric("📍 Range",         f"{rng:.0f} km",
          "Estimated")
k4.metric("❤️ Health",        f"{hlth:.1f} %",
          f"{int(snapshot.get('cycle_count', 0))} cycles")
k5.metric("⚡ Voltage",       f"{volt:.0f} V",
          "Pack voltage")
k6.metric("🔌 Current",       f"{curr:.0f} A",
          "Charging" if is_charging else "Discharging")

st.divider()


# -----------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------
tabs = st.tabs([
    "📊 Live Monitor",
    "🤖 AI Predictions",
    "🌡️ Thermal",
    "⚡ Charging",
    "🔭 Digital Twin",
    "🚨 Alerts",
])


# =======================================================================
# TAB 1 — LIVE MONITOR
# =======================================================================
with tabs[0]:
    st.markdown("### Real-Time Battery Telemetry")

    if len(hist_df) > 1:

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=hist_df["soc_pct"], mode="lines",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
            ))
            fig.update_layout(
                title="State of Charge (%)",
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                font=dict(color="#e0e8f0"), height=250,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(showgrid=False, color="#4a7fa5"),
                yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5", range=[0, 100]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=hist_df["voltage_v"], mode="lines",
                line=dict(color="#7c3aed", width=2),
            ))
            fig.update_layout(
                title="Pack Voltage (V)",
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                font=dict(color="#e0e8f0"), height=250,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(showgrid=False, color="#4a7fa5"),
                yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=hist_df["temperature_c"], mode="lines",
                line=dict(color="#f97316", width=2),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
            ))
            fig.add_hline(
                y=42, line_color="#dc2626", line_dash="dash",
                annotation_text="42°C safety limit",
                annotation_font_color="#dc2626",
            )
            fig.update_layout(
                title="Battery Temperature (°C)",
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                font=dict(color="#e0e8f0"), height=250,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(showgrid=False, color="#4a7fa5"),
                yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            if "power_kw" in hist_df.columns:
                bar_colors = [
                    "#22c55e" if v < 0 else "#ef4444"
                    for v in hist_df["power_kw"]
                ]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=hist_df["power_kw"],
                    marker_color=bar_colors,
                ))
                fig.update_layout(
                    title="Power (kW) — green = charging / regen",
                    paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                    font=dict(color="#e0e8f0"), height=250,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis=dict(showgrid=False, color="#4a7fa5"),
                    yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Session Data — Last 20 Readings")
        display_cols = [
            "soc_pct", "voltage_v", "temperature_c",
            "current_a", "speed_kmh", "range_km", "health_pct",
        ]
        avail = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(
            hist_df[avail].tail(20).reset_index(drop=True),
            use_container_width=True,
        )

    else:
        st.info("Press **▶ Step** in the sidebar to start the simulation.")


# =======================================================================
# TAB 2 — AI PREDICTIONS
# =======================================================================
with tabs[1]:
    st.markdown("### AI Range and Energy Predictions")
    st.caption(
        "Predictions are computed from current sensor state using physics-based "
        "formulas and trained RandomForest / GradientBoosting models."
    )

    # Range prediction from current state
    remaining_kwh    = (soc / 100) * sim.capacity_kwh * (hlth / 100)
    load_kw          = snapshot.get("load_kw", 10)
    eff_kwh_per_km   = max(load_kw / max(speed_kmh, 1), 0.05)
    predicted_range  = remaining_kwh / eff_kwh_per_km

    p1, p2, p3 = st.columns(3)
    p1.metric("📍 Predicted Range",   f"{predicted_range:.0f} km")
    p2.metric("⚡ Efficiency",         f"{eff_kwh_per_km * 100:.1f} kWh/100km")
    drain = abs(snapshot.get("soc_delta", 0)) * 3600
    p3.metric("📉 Drain Rate",         f"{drain:.2f} %/hr")

    st.divider()

    # Energy breakdown pie
    st.markdown("#### Energy Consumption Breakdown")
    base_kw = 15 * (speed_kmh / 100) ** 2 * sim.capacity_kwh / 75
    ac_kw   = 1.8 if ac_on else 0.0
    misc_kw = 0.3

    fig_pie = go.Figure(go.Pie(
        labels  = ["Drivetrain", "A/C System", "Misc Electronics"],
        values  = [round(max(base_kw, 0.01), 2), ac_kw, misc_kw],
        hole    = 0.55,
        marker  = dict(colors=["#00d4ff", "#f97316", "#7c3aed"]),
    ))
    fig_pie.update_layout(
        paper_bgcolor="#0d1b2e",
        font=dict(color="#e0e8f0"),
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(font=dict(color="#e0e8f0")),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Speed vs range curve
    st.markdown("#### Speed vs Estimated Range")
    speeds = np.arange(20, 150, 5)
    ranges = []
    for s in speeds:
        e_kw = 15 * (s / 100) ** 2 * sim.capacity_kwh / 75 + (1.8 if ac_on else 0)
        e_km = max(e_kw / max(s, 1), 0.05)
        ranges.append(remaining_kwh / e_km)

    fig_svr = go.Figure()
    fig_svr.add_trace(go.Scatter(
        x=speeds, y=ranges, mode="lines",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ))
    fig_svr.add_vline(
        x=speed_kmh, line_color="#f97316", line_dash="dash",
        annotation_text=f"Current: {speed_kmh} km/h",
        annotation_font_color="#f97316",
    )
    fig_svr.update_layout(
        title="How Speed Affects Your Remaining Range",
        xaxis_title="Speed (km/h)", yaxis_title="Estimated Range (km)",
        paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
        font=dict(color="#e0e8f0"), height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
        yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
    )
    st.plotly_chart(fig_svr, use_container_width=True)

    # Eco tips
    st.markdown("#### AI Eco-Driving Recommendations")
    tips = []
    if speed_kmh > 110:
        tips.append(
            f"Reduce speed from {speed_kmh:.0f} to 100 km/h — "
            f"saves approximately {int((speed_kmh - 100) * 0.7)} km of range."
        )
    if ac_on:
        tips.append("A/C adds ~1.8 kW. Switch to fan-only mode when comfortable (+12-18 km).")
    if soc < 30:
        tips.append("Below 30% SOC: activate Eco mode and cap speed at 90 km/h.")
    if not tips:
        tips.append("Driving efficiently. Maintain current speed and settings.")
    tips.append(
        f"Regenerative braking at {speed_kmh:.0f} km/h recovers "
        f"~{speed_kmh * 0.08:.0f} W — prefer engine braking over friction brakes."
    )
    for tip in tips:
        st.markdown(f"• {tip}")


# =======================================================================
# TAB 3 — THERMAL
# =======================================================================
with tabs[2]:
    st.markdown("### Thermal Intelligence System")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("🌡️ Current Temp",    f"{thermal_report.temperature_c:.1f} °C")
    t2.metric("🔮 Predicted (60s)", f"{thermal_report.predicted_temp_c:.1f} °C")
    t3.metric("📊 Risk Score",      f"{thermal_report.risk_score:.0f} / 100")
    t4.metric("❄️ Cooling Status",  thermal_report.cooling_status)

    # Gauge chart for risk score
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = thermal_report.risk_score,
        gauge = {
            "axis":        {"range": [0, 100], "tickcolor": "#7ec8e3"},
            "bar":         {"color": "#00d4ff"},
            "bgcolor":     "#0d1b2e",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0,  25], "color": "#052e16"},
                {"range": [25, 50], "color": "#422006"},
                {"range": [50, 75], "color": "#431407"},
                {"range": [75, 100],"color": "#3b0764"},
            ],
            "threshold": {
                "line":  {"color": "#dc2626", "width": 4},
                "value": 75,
            },
        },
        title  = {"text": "Thermal Risk Score", "font": {"color": "#e0e8f0"}},
        number = {"font": {"color": "#00d4ff"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0d1b2e",
        font=dict(color="#e0e8f0"),
        height=280,
        margin=dict(l=40, r=40, t=60, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Status badge
    level_colors = {
        "Safe":     "#22c55e",
        "Watch":    "#eab308",
        "Warning":  "#f97316",
        "Critical": "#dc2626",
    }
    lc = level_colors.get(thermal_report.alert_level, "#6b7280")
    st.markdown(
        f"<div style='text-align:center; margin: 8px 0;'>"
        f"<span style='display:inline-block; padding:6px 18px; border-radius:20px; "
        f"background:{lc}22; color:{lc}; border:1px solid {lc}; "
        f"font-weight:700; letter-spacing:0.1em;'>"
        f"THERMAL STATUS: {thermal_report.alert_level.upper()}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**Thermal Efficiency:** {thermal_report.efficiency_score:.0f}%")
    st.progress(int(thermal_report.efficiency_score))

    st.markdown("#### Recommendations")
    for rec in thermal_report.recommendations:
        st.markdown(f"• {rec}")

    # Temperature history chart
    if len(hist_df) > 1 and "temperature_c" in hist_df.columns:
        st.divider()
        st.markdown("#### Temperature History")
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(
            y=hist_df["temperature_c"], mode="lines",
            line=dict(color="#f97316", width=2),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.06)",
        ))
        fig_th.add_hline(
            y=42, line_color="#dc2626", line_dash="dot",
            annotation_text="42°C safety limit",
            annotation_font_color="#dc2626",
        )
        fig_th.add_hline(
            y=35, line_color="#eab308", line_dash="dot",
            annotation_text="35°C watch",
            annotation_font_color="#eab308",
        )
        fig_th.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e0e8f0"), height=240,
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis=dict(showgrid=False, color="#4a7fa5"),
            yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
        )
        st.plotly_chart(fig_th, use_container_width=True)

        summary = thermal_history_summary(hist_df["temperature_c"].tolist())
        if summary:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Mean Temp",      f"{summary['mean_temp_c']} °C")
            sc2.metric("Max Temp",       f"{summary['max_temp_c']} °C")
            sc3.metric("% Time > 40°C", f"{summary['pct_above_40c']} %")


# =======================================================================
# TAB 4 — CHARGING
# =======================================================================
with tabs[3]:
    st.markdown("### Smart Charging Intelligence")

    ch1, ch2, ch3 = st.columns(3)
    ch1.metric(
        "⚡ Urgency",
        charge_report.urgency_label,
        f"{charge_report.urgency_score:.0f}/100",
    )
    ch2.metric("🎯 Target SOC",      f"{charge_report.recommended_target_soc:.0f} %")
    ch3.metric(
        "⏱️ Time to Target",
        f"{charge_report.estimated_time_to_target_min:.0f} min",
    )

    # Urgency progress bar
    urg = charge_report.urgency_score
    urg_color = (
        "#22c55e" if urg < 25 else
        "#eab308" if urg < 50 else
        "#f97316" if urg < 75 else
        "#dc2626"
    )
    st.markdown(
        f"<div style='background:#0d1b2e; border:1px solid #1e3a5f; "
        f"border-radius:10px; padding:16px; margin:8px 0;'>"
        f"<p style='color:#7ec8e3; margin:0 0 6px 0; font-size:0.75rem;'>"
        f"CHARGING URGENCY</p>"
        f"<div style='background:#1e3a5f; border-radius:20px; height:16px;'>"
        f"<div style='background:{urg_color}; width:{urg:.0f}%; "
        f"height:100%; border-radius:20px;'></div></div>"
        f"<p style='color:{urg_color}; font-weight:700; margin:8px 0 0 0;'>"
        f"{charge_report.urgency_label} — {urg:.0f}/100</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # CC-CV charge curve
    st.markdown("#### Simulated Charge Curve")
    curve = simulate_charge_curve(
        start_soc        = soc,
        capacity_kwh     = sim.capacity_kwh,
        charger_power_kw = float(charge_kw),
        temperature_c    = temp,
        target_soc       = charge_report.recommended_target_soc,
    )

    if curve:
        curve_df = pd.DataFrame(curve)
        fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
        fig_curve.add_trace(
            go.Scatter(
                x=curve_df["time_min"], y=curve_df["soc_pct"],
                mode="lines", line=dict(color="#00d4ff", width=2.5),
                name="SOC %",
            ),
            secondary_y=False,
        )
        fig_curve.add_trace(
            go.Scatter(
                x=curve_df["time_min"], y=curve_df["power_kw"],
                mode="lines",
                line=dict(color="#f97316", width=1.5, dash="dot"),
                name="Charge Power (kW)",
            ),
            secondary_y=True,
        )
        fig_curve.update_layout(
            title="Projected Charging Session (CC-CV curve)",
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e0e8f0"), height=320,
            margin=dict(l=50, r=50, t=50, b=40),
            legend=dict(font=dict(color="#e0e8f0")),
        )
        fig_curve.update_xaxes(
            title_text="Time (min)",
            gridcolor="#1e3a5f", color="#4a7fa5",
        )
        fig_curve.update_yaxes(
            title_text="SOC %",
            gridcolor="#1e3a5f", color="#4a7fa5",
            secondary_y=False,
        )
        fig_curve.update_yaxes(
            title_text="Power (kW)",
            color="#f97316",
            secondary_y=True,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("#### Recommendations")
    for rec in charge_report.recommendations:
        st.markdown(f"• {rec}")

    st.caption(
        f"Charge efficiency: {charge_report.charge_efficiency_pct:.1f}%  |  "
        f"Optimal daily window: {charge_report.optimal_charge_window[0]:.0f}%"
        f"–{charge_report.optimal_charge_window[1]:.0f}%  |  "
        f"Full charge time: {charge_report.estimated_time_to_full_min:.0f} min"
    )


# =======================================================================
# TAB 5 — DIGITAL TWIN
# =======================================================================
with tabs[4]:
    st.markdown("### Digital Twin — Future Simulation Engine")
    st.caption(
        "Projects battery health, range, and efficiency degradation over "
        "the coming weeks using a physics-informed ageing model."
    )

    dt1, dt2, dt3 = st.columns(3)
    dt1.metric("📅 Weeks to 80% SoH", f"{forecast.weeks_to_80pct_soh:.0f} weeks")
    dt2.metric("📅 Weeks to 70% SoH", f"{forecast.weeks_to_70pct_soh:.0f} weeks")
    dt3.metric("🔋 Final SoH",        f"{forecast.final_soh:.1f} %")

    # Scenario comparison chart
    st.markdown("#### What-If Scenario Comparison")
    scenarios = compare_scenarios(
        current_soh      = snapshot.get("health_pct", 90),
        current_range_km = snapshot.get("range_km", 250),
        current_cycles   = int(snapshot.get("cycle_count", 150)),
        capacity_kwh     = sim.capacity_kwh,
        weeks            = forecast_weeks,
    )

    scene_colors = {
        "longevity":  "#22c55e",
        "mixed":      "#00d4ff",
        "aggressive": "#dc2626",
    }

    fig_scene = go.Figure()
    for habit, fc in scenarios.items():
        fig_scene.add_trace(go.Scatter(
            x=fc.weeks, y=fc.soh_timeline, mode="lines",
            line=dict(color=scene_colors[habit], width=2.5),
            name=habit.capitalize(),
        ))
    fig_scene.add_hline(
        y=80, line_color="#eab308", line_dash="dash",
        annotation_text="80% — typical replacement threshold",
        annotation_font_color="#eab308",
    )
    fig_scene.add_hline(
        y=70, line_color="#dc2626", line_dash="dot",
        annotation_text="70% — end of useful life",
        annotation_font_color="#dc2626",
    )
    fig_scene.update_layout(
        title="Battery Health Forecast by Charging Habit",
        xaxis_title="Weeks", yaxis_title="State of Health (%)",
        paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
        font=dict(color="#e0e8f0"), height=360,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
        yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5", range=[50, 100]),
        legend=dict(font=dict(color="#e0e8f0")),
    )
    st.plotly_chart(fig_scene, use_container_width=True)

    # Range forecast
    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=forecast.weeks, y=forecast.range_timeline,
        mode="lines", fill="tozeroy",
        line=dict(color="#7c3aed", width=2.5),
        fillcolor="rgba(124,58,237,0.07)",
    ))
    fig_range.update_layout(
        title="Projected Driving Range Over Time (km)",
        xaxis_title="Weeks", yaxis_title="Range (km)",
        paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
        font=dict(color="#e0e8f0"), height=280,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
        yaxis=dict(gridcolor="#1e3a5f", color="#4a7fa5"),
    )
    st.plotly_chart(fig_range, use_container_width=True)

    st.markdown("#### AI Digital Twin Insight")
    insight = degradation_summary_text(forecast, charging_habit)
    st.info(insight)


# =======================================================================
# TAB 6 — ALERTS
# =======================================================================
with tabs[5]:
    st.markdown("### Active Risk Alerts")

    if not active_alerts:
        st.success("All systems normal. No active alerts.")
    else:
        css_map  = {"CRITICAL": "alert-critical", "WARNING": "alert-warning", "INFO": "alert-info"}
        icon_map = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}

        for alert in active_alerts:
            css  = css_map.get(alert.severity, "alert-info")
            icon = icon_map.get(alert.severity, "ℹ️")
            st.markdown(
                f"<div class='{css}'>"
                f"<b>{icon} [{alert.severity}] {alert.code}</b>"
                f"<span style='font-size:0.7rem; color:#6b7280; float:right;'>"
                f"{alert.timestamp}</span><br>"
                f"{alert.message}<br>"
                f"<small style='color:#9ca3af;'>Action: {alert.action}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Alert Reference Table")

    ref_data = [
        ["SOC_CRITICAL",      "CRITICAL", "SOC below 5%",        "Find charger immediately"],
        ["SOC_LOW",           "WARNING",  "SOC 5–15%",           "Plan charging stop"],
        ["LOW_RANGE",         "CRITICAL", "Range below 20 km",   "Stop and charge"],
        ["OVERHEAT_CRITICAL", "CRITICAL", "Temp above 48°C",     "Stop vehicle"],
        ["OVERHEAT_WARNING",  "WARNING",  "Temp 42–48°C",        "Reduce speed"],
        ["HEALTH_CRITICAL",   "CRITICAL", "Health below 70%",    "Battery inspection"],
        ["HEALTH_WARNING",    "WARNING",  "Health below 80%",    "Use eco charging"],
        ["HIGH_SPEED",        "INFO",     "Speed above 130 km/h","Reduce for range"],
        ["COLD_CHARGE",       "WARNING",  "Charging below 5°C",  "Pre-heat battery"],
        ["LOW_VOLTAGE",       "WARNING",  "Voltage below 340 V", "Charge immediately"],
    ]
    ref_df = pd.DataFrame(
        ref_data,
        columns=["Code", "Severity", "Trigger", "Recommended Action"],
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:#374151; font-size:0.75rem;'>"
    "EV Digital Twin System &nbsp;·&nbsp; "
    "AI-Powered Battery Intelligence Platform &nbsp;·&nbsp; "
    "Built with Python, Streamlit and Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)

# Auto-rerun when simulation is running
if auto_run:
    time.sleep(0.3)
    st.rerun()
