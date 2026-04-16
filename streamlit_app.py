"""
AWS1 Gilgit Weather Forecast Dashboard
========================================
Streamlit frontend for the weather forecasting system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import xgboost as xgb
import os
from datetime import datetime, timedelta
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engine import engineer_features, get_feature_cols

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AWS1 Gilgit Weather Forecast",
    page_icon  = "🌤",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .confidence-high   { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low    { color: #e74c3c; font-weight: 600; }
    .forecast-table {
        border-radius: 10px;
        overflow: hidden;
    }
    .stAlert { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
TARGETS   = ["AirTC_Avg", "RH_Avg", "B_Pressure_Avg", "WindSpeed_Avg"]
HORIZONS  = ["+1hr", "+3hr", "+6hr", "+12hr", "+24hr"]

TARGET_LABELS = {
    "AirTC_Avg"     : "Temperature",
    "RH_Avg"        : "Humidity",
    "B_Pressure_Avg": "Pressure",
    "WindSpeed_Avg" : "Wind Speed",
}
TARGET_UNITS = {
    "AirTC_Avg"     : "°C",
    "RH_Avg"        : "%",
    "B_Pressure_Avg": "hPa",
    "WindSpeed_Avg" : "m/s",
}
TARGET_ICONS = {
    "AirTC_Avg"     : "🌡",
    "RH_Avg"        : "💧",
    "B_Pressure_Avg": "📊",
    "WindSpeed_Avg" : "💨",
}
CONFIDENCE_MAP = {
    "AirTC_Avg"     : {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Medium"},
    "RH_Avg"        : {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Low"},
    "B_Pressure_Avg": {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Medium"},
    "WindSpeed_Avg" : {"+1hr":"Medium", "+3hr":"Low", "+6hr":"Low",
                       "+12hr":"Low", "+24hr":"Low"},
}
HORIZON_HOURS = {
    "+1hr":1, "+3hr":3, "+6hr":6, "+12hr":12, "+24hr":24
}

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models    = {}
    rain_clfs = {}
    scalers_y = {}

    scaler_X = joblib.load(f"{MODEL_DIR}/scaler_X.pkl")

    for target in TARGETS:
        models[target]    = {}
        scalers_y[target] = joblib.load(f"{MODEL_DIR}/scaler_y_{target}.pkl")
        for horizon in HORIZONS:
            m = xgb.XGBRegressor()
            m.load_model(f"{MODEL_DIR}/xgb_{target}_{horizon}.json")
            models[target][horizon] = m

    for horizon in HORIZONS:
        clf = xgb.XGBClassifier()
        clf.load_model(f"{MODEL_DIR}/xgb_rain_{horizon}.json")
        rain_clfs[horizon] = clf

    return scaler_X, models, rain_clfs, scalers_y

# ── Generate Forecast ─────────────────────────────────────────────────────────
def generate_forecast(df, scaler_X, models, rain_clfs):
    df_eng       = engineer_features(df)
    feature_cols = get_feature_cols(df_eng)
    X            = scaler_X.transform(df_eng[feature_cols])
    X_last       = X[[-1]]
    last_row     = df_eng.iloc[-1]

    current = {
        "timestamp"   : last_row["TIMESTAMP"],
        "AirTC_Avg"   : float(last_row["AirTC_Avg"]),
        "RH_Avg"      : float(last_row["RH_Avg"]),
        "B_Pressure_Avg": float(last_row["B_Pressure_Avg"]),
        "WindSpeed_Avg" : float(last_row["WindSpeed_Avg"]),
        "Rain_mm_Tot"   : float(last_row["Rain_mm_Tot"]),
        "BattV_Avg"     : float(last_row["BattV_Avg"]),
    }

    forecasts = []
    for horizon in HORIZONS:
        row = {"horizon": horizon}
        for target in TARGETS:
            pred = float(models[target][horizon].predict(X_last)[0])
            if target == "RH_Avg":
                pred = np.clip(pred, 0, 100)
            elif target == "WindSpeed_Avg":
                pred = np.clip(pred, 0, 50)
            elif target == "B_Pressure_Avg":
                pred = np.clip(pred, 800, 900)
            row[target] = round(pred, 1)

        rain_prob = float(rain_clfs[horizon].predict_proba(X_last)[0][1])
        row["rain_probability"] = round(rain_prob * 100, 1)
        forecasts.append(row)

    return current, pd.DataFrame(forecasts)

# ── Confidence Badge ──────────────────────────────────────────────────────────
def confidence_badge(level):
    colors = {"High":"🟢", "Medium":"🟡", "Low":"🔴"}
    return f"{colors.get(level,'⚪')} {level}"

# ── Weather Icon ──────────────────────────────────────────────────────────────
def weather_icon(temp, rh, rain_prob, wind):
    if rain_prob > 60:
        return "⛈️" if wind > 2 else "🌧️"
    elif rain_prob > 30:
        return "🌦️"
    elif rh > 80:
        return "☁️"
    elif rh > 60:
        return "⛅"
    elif temp > 35:
        return "☀️🔥"
    else:
        return "☀️"

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    '<div class="main-header">🌤 AWS1 Gilgit Weather Forecast</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Automatic Weather Station — Gilgit, Pakistan | ' +
    'Altitude: ~1500m | 24-Hour Forecast System</div>',
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/200px-Flag_of_Pakistan.svg.png",
        width=100
    )
    st.title("⚙️ Settings")

    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Upload CSV file", "Use sample data"],
        index=1
    )

    uploaded_file = None
    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader(
            "Upload your .csv sensor data file",
            type=["csv"],
            help="Upload cleaned sensor data with required columns"
        )

    st.markdown("---")
    st.markdown("### 📊 Display Settings")
    show_confidence = st.checkbox("Show confidence levels", value=True)
    show_charts     = st.checkbox("Show forecast charts",   value=True)
    show_raw_data   = st.checkbox("Show raw data table",    value=False)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model:** XGBoost ensemble
    **Training:** May 2024 – Aug 2025
    **Station:** AWS1 Gilgit (CR350)
    **Interval:** 10-minute
    **Horizons:** 1h, 3h, 6h, 12h, 24h
    """)

    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    perf_data = {
        "Variable"  : ["Temperature", "Humidity", "Pressure", "Wind", "Rain"],
        "R² (+1hr)" : ["0.991", "0.964", "0.995", "0.559", "F1:0.509"],
        "R² (+24hr)": ["0.872", "0.463", "0.769", "0.163", "F1:0.524"],
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

# ── Load Models ───────────────────────────────────────────────────────────────
with st.spinner("Loading forecast models..."):
    try:
        scaler_X, models, rain_clfs, scalers_y = load_models()
        st.sidebar.success("✓ Models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    """Load the most recent data from engineered CSV."""
    # Try to load recent data
    paths = [
        "data_cleaned.csv",
        "combined_clean.csv",
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
            df = df.sort_values("TIMESTAMP").tail(500)
            return df
    return None

# Get data
df_input = None

if data_source == "Upload CSV file" and uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        df_input["TIMESTAMP"] = pd.to_datetime(df_input["TIMESTAMP"])
        df_input = df_input.sort_values("TIMESTAMP")
        st.sidebar.success(f"✓ Loaded {len(df_input):,} rows")
    except Exception as e:
        st.error(f"Error loading file: {e}")

elif data_source == "Use sample data":
    df_input = load_sample_data()
    if df_input is not None:
        st.sidebar.info(f"Using last {len(df_input)} rows of station data")

# ── Generate Forecast ─────────────────────────────────────────────────────────
if df_input is not None and len(df_input) >= 50:
    try:
        with st.spinner("Generating forecast..."):
            current, forecast_df = generate_forecast(
                df_input, scaler_X, models, rain_clfs
            )

        # ── Current Conditions ────────────────────────────────────────────────
        st.markdown("## 📍 Current Conditions")
        ts = pd.to_datetime(current["timestamp"])
        st.caption(
            f"Last reading: {ts.strftime('%B %d, %Y at %H:%M')} | "
            f"Battery: {current['BattV_Avg']:.1f}V"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "🌡 Temperature",
                f"{current['AirTC_Avg']:.1f}°C",
            )
        with col2:
            st.metric(
                "💧 Humidity",
                f"{current['RH_Avg']:.1f}%",
            )
        with col3:
            st.metric(
                "📊 Pressure",
                f"{current['B_Pressure_Avg']:.0f} hPa",
            )
        with col4:
            st.metric(
                "💨 Wind Speed",
                f"{current['WindSpeed_Avg']:.1f} m/s",
            )
        with col5:
            st.metric(
                "🌧 Rain",
                f"{current['Rain_mm_Tot']:.1f} mm",
            )

        st.markdown("---")

        # ── Forecast Table ────────────────────────────────────────────────────
        st.markdown("## 🔮 Weather Forecast")

        # Build display table
        table_rows = []
        for _, row in forecast_df.iterrows():
            h = row["horizon"]
            future_ts = ts + timedelta(hours=HORIZON_HOURS[h])
            icon = weather_icon(
                row["AirTC_Avg"], row["RH_Avg"],
                row["rain_probability"], row["WindSpeed_Avg"]
            )
            display_row = {
                "Time"        : future_ts.strftime("%H:%M"),
                "Horizon"     : h,
                "Weather"     : icon,
                "Temp (°C)"   : f"{row['AirTC_Avg']:.1f}",
                "RH (%)"      : f"{row['RH_Avg']:.1f}",
                "Pressure (hPa)": f"{row['B_Pressure_Avg']:.0f}",
                "Wind (m/s)"  : f"{row['WindSpeed_Avg']:.1f}",
                "Rain Prob"   : f"{row['rain_probability']:.0f}%",
            }
            if show_confidence:
                display_row["Temp Conf"]  = confidence_badge(CONFIDENCE_MAP["AirTC_Avg"][h])
                display_row["Wind Conf"]  = confidence_badge(CONFIDENCE_MAP["WindSpeed_Avg"][h])
            table_rows.append(display_row)

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index       = True,
            use_container_width = True,
        )

        # Rain alert
        max_rain_prob = forecast_df["rain_probability"].max()
        if max_rain_prob > 60:
            st.warning(
                f"⛈️ High rain probability detected: "
                f"{max_rain_prob:.0f}% in next 24 hours"
            )
        elif max_rain_prob > 30:
            st.info(
                f"🌦️ Moderate rain possibility: "
                f"{max_rain_prob:.0f}% in next 24 hours"
            )
        else:
            st.success("☀️ Low rain probability for next 24 hours")

        st.markdown("---")

        # ── Forecast Charts ───────────────────────────────────────────────────
        if show_charts:
            st.markdown("## 📈 Forecast Charts")

            # Prepare chart data
            future_times = [
                ts + timedelta(hours=HORIZON_HOURS[h])
                for h in HORIZONS
            ]

            # Current + forecast combined
            all_times  = [ts] + future_times
            all_temp   = [current["AirTC_Avg"]]   + forecast_df["AirTC_Avg"].tolist()
            all_rh     = [current["RH_Avg"]]       + forecast_df["RH_Avg"].tolist()
            all_press  = [current["B_Pressure_Avg"]] + forecast_df["B_Pressure_Avg"].tolist()
            all_wind   = [current["WindSpeed_Avg"]] + forecast_df["WindSpeed_Avg"].tolist()
            all_rain_p = [0] + forecast_df["rain_probability"].tolist()

            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "🌡 Temperature (°C)",
                    "💧 Humidity (%)",
                    "📊 Pressure (hPa)",
                    "💨 Wind Speed (m/s)",
                    "🌧 Rain Probability (%)",
                    ""
                ),
                vertical_spacing   = 0.12,
                horizontal_spacing = 0.08,
            )

            # Color scheme
            colors = {
                "temp"  : "#e74c3c",
                "rh"    : "#3498db",
                "press" : "#9b59b6",
                "wind"  : "#27ae60",
                "rain"  : "#2980b9",
            }

            # Temperature
            fig.add_trace(go.Scatter(
                x=all_times, y=all_temp,
                mode="lines+markers",
                name="Temperature",
                line=dict(color=colors["temp"], width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.1)"
            ), row=1, col=1)

            # Humidity
            fig.add_trace(go.Scatter(
                x=all_times, y=all_rh,
                mode="lines+markers",
                name="Humidity",
                line=dict(color=colors["rh"], width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.1)"
            ), row=1, col=2)

            # Pressure
            fig.add_trace(go.Scatter(
                x=all_times, y=all_press,
                mode="lines+markers",
                name="Pressure",
                line=dict(color=colors["press"], width=3),
                marker=dict(size=8),
            ), row=2, col=1)

            # Wind
            fig.add_trace(go.Bar(
                x=all_times, y=all_wind,
                name="Wind Speed",
                marker_color=colors["wind"],
                opacity=0.8,
            ), row=2, col=2)

            # Rain probability
            fig.add_trace(go.Bar(
                x=all_times, y=all_rain_p,
                name="Rain Probability",
                marker_color=[
                    "#e74c3c" if p > 60 else
                    "#f39c12" if p > 30 else
                    "#27ae60"
                    for p in all_rain_p
                ],
                opacity=0.85,
            ), row=3, col=1)

            # Add 50% rain threshold line
            fig.add_hline(
                y=50, row=3, col=1,
                line_dash="dash",
                line_color="red",
                annotation_text="50% threshold"
            )

            fig.update_layout(
                height          = 750,
                showlegend      = False,
                title_text      = f"AWS1 Gilgit — 24 Hour Forecast from {ts.strftime('%b %d, %H:%M')}",
                title_font_size = 16,
                paper_bgcolor   = "white",
                plot_bgcolor    = "#f8f9fa",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Historical Data Charts ────────────────────────────────────────────
        st.markdown("## 📉 Recent Station Data (Last 48 Hours)")

        # Show last 48 hours of historical data
        df_recent = df_input.tail(288).copy()
        if "AirTC_Avg" in df_recent.columns:
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Temperature (°C)", "Humidity (%)",
                    "Pressure (hPa)",   "Wind Speed (m/s)"
                ),
                vertical_spacing   = 0.15,
                horizontal_spacing = 0.08,
            )

            plot_vars = [
                ("AirTC_Avg",      "#e74c3c", 1, 1),
                ("RH_Avg",         "#3498db", 1, 2),
                ("B_Pressure_Avg", "#9b59b6", 2, 1),
                ("WindSpeed_Avg",  "#27ae60", 2, 2),
            ]

            for col_name, color, row, col in plot_vars:
                if col_name in df_recent.columns:
                    fig2.add_trace(go.Scatter(
                        x    = df_recent["TIMESTAMP"],
                        y    = df_recent[col_name],
                        mode = "lines",
                        name = col_name,
                        line = dict(color=color, width=1.5),
                    ), row=row, col=col)

            fig2.update_layout(
                height        = 500,
                showlegend    = False,
                title_text    = "Last 48 Hours — Sensor Readings",
                paper_bgcolor = "white",
                plot_bgcolor  = "#f8f9fa",
            )
            fig2.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            fig2.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Raw Data ──────────────────────────────────────────────────────────
        if show_raw_data:
            st.markdown("## 🗄️ Raw Forecast Data")
            st.dataframe(forecast_df, use_container_width=True)

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#888; font-size:0.85rem;'>
        AWS1 Gilgit Weather Forecast System |
        Model trained on May 2024 – Aug 2025 data |
        XGBoost Multi-Horizon Forecasting |
        Developed for BIS Internship Project
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        st.exception(e)

else:
    # Welcome screen
    st.info(
        "👈 Please select a data source from the sidebar to generate a forecast."
    )
    st.markdown("""
    ## 🚀 Getting Started

    This dashboard provides **24-hour weather forecasts** for Gilgit, Pakistan
    using machine learning models trained on AWS1 station data.

    ### How to use:
    1. **Upload CSV** — Upload your cleaned sensor data file
    2. **Use sample data** — Use the latest station readings

    ### What you get:
    - ✅ Temperature forecast (1hr to 24hr)
    - ✅ Humidity forecast (1hr to 24hr)
    - ✅ Pressure forecast (1hr to 24hr)
    - ✅ Wind speed forecast (1hr only reliable)
    - ✅ Rain probability (1hr to 24hr)
    - ✅ Confidence indicators for each forecast
    - ✅ Interactive charts
    - ✅ Last 48 hours of station data

    ### Model Performance:
    | Variable | R² at +1hr | R² at +24hr |
    |----------|-----------|------------|
    | Temperature | 0.991 | 0.872 |
    | Humidity | 0.964 | 0.463 |
    | Pressure | 0.995 | 0.769 |
    | Wind | 0.559 | 0.163 |
    | Rain (F1) | 0.509 | 0.524 |
    """)
