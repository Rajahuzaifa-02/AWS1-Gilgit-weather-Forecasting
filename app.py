"""
AWS1 Gilgit Weather Forecast — FastAPI Backend
================================================
Endpoints:
  GET /           → API info
  GET /health     → system health check
  POST /predict   → generate forecast from sensor readings
  GET /forecast   → forecast using latest data from CSV
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
from datetime import datetime

from feature_engine import engineer_features, get_feature_cols

# ── Initialize app ────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AWS1 Gilgit Weather Forecast API",
    description = "24-hour weather forecasting for Gilgit, Pakistan",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
TARGETS    = ["AirTC_Avg", "RH_Avg", "B_Pressure_Avg", "WindSpeed_Avg"]
HORIZONS   = ["+1hr", "+3hr", "+6hr", "+12hr", "+24hr"]

TARGET_LABELS = {
    "AirTC_Avg"     : "Temperature (°C)",
    "RH_Avg"        : "Humidity (%)",
    "B_Pressure_Avg": "Pressure (hPa)",
    "WindSpeed_Avg" : "Wind Speed (m/s)",
}

CONFIDENCE = {
    "AirTC_Avg"     : {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Medium"},
    "RH_Avg"        : {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Low"},
    "B_Pressure_Avg": {"+1hr":"High", "+3hr":"High", "+6hr":"High",
                       "+12hr":"Medium", "+24hr":"Medium"},
    "WindSpeed_Avg" : {"+1hr":"Medium", "+3hr":"Low", "+6hr":"Low",
                       "+12hr":"Low", "+24hr":"Low"},
}

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading models...")
models     = {}
rain_clfs  = {}
scalers_y  = {}

scaler_X = joblib.load(f"{MODEL_DIR}/scaler_X.pkl")

for target in TARGETS:
    models[target] = {}
    scalers_y[target] = joblib.load(f"{MODEL_DIR}/scaler_y_{target}.pkl")
    for horizon in HORIZONS:
        m = xgb.XGBRegressor()
        m.load_model(f"{MODEL_DIR}/xgb_{target}_{horizon}.json")
        models[target][horizon] = m

for horizon in HORIZONS:
    clf = xgb.XGBClassifier()
    clf.load_model(f"{MODEL_DIR}/xgb_rain_{horizon}.json")
    rain_clfs[horizon] = clf

print(f"✓ Loaded {len(TARGETS) * len(HORIZONS)} regression models")
print(f"✓ Loaded {len(HORIZONS)} rain classifiers")

# ── Request/Response Models ───────────────────────────────────────────────────
class SensorReading(BaseModel):
    timestamp        : str
    AirTC_Avg        : float
    AirTC_Min        : float
    AirTC_Max        : float
    RH_Avg           : float
    RH_Min           : float
    RH_Max           : float
    B_Pressure_Avg   : float
    WindSpeed_Avg    : float
    WindSpeed_Min    : float
    WindSpeed_Max    : float
    WindDir_D1_WVT   : float
    WindDir_SD1_WVT  : float
    Rain_mm_Tot      : float
    PTemp_C_Avg      : float
    BattV_Avg        : float
    WindSpeed_S_WVT  : Optional[float] = 0.0

class ForecastRequest(BaseModel):
    # Last 48 hours of sensor readings (288 rows minimum)
    # Each row is a SensorReading
    readings: list[SensorReading]

# ── Helper Functions ──────────────────────────────────────────────────────────
def readings_to_df(readings: list) -> pd.DataFrame:
    """Convert list of SensorReading to DataFrame."""
    rows = []
    for r in readings:
        rows.append({
            'TIMESTAMP'      : pd.to_datetime(r.timestamp),
            'AirTC_Avg'      : r.AirTC_Avg,
            'AirTC_Min'      : r.AirTC_Min,
            'AirTC_Max'      : r.AirTC_Max,
            'RH_Avg'         : r.RH_Avg,
            'RH_Min'         : r.RH_Min,
            'RH_Max'         : r.RH_Max,
            'B_Pressure_Avg' : r.B_Pressure_Avg,
            'WindSpeed_Avg'  : r.WindSpeed_Avg,
            'WindSpeed_Min'  : r.WindSpeed_Min,
            'WindSpeed_Max'  : r.WindSpeed_Max,
            'WindDir_D1_WVT' : r.WindDir_D1_WVT,
            'WindDir_SD1_WVT': r.WindDir_SD1_WVT,
            'Rain_mm_Tot'    : r.Rain_mm_Tot,
            'PTemp_C_Avg'    : r.PTemp_C_Avg,
            'BattV_Avg'      : r.BattV_Avg,
            'WindSpeed_S_WVT': r.WindSpeed_S_WVT,
        })
    return pd.DataFrame(rows)


def generate_forecast(df: pd.DataFrame) -> dict:
    """Run feature engineering and generate all forecasts."""

    # Engineer features
    df_eng = engineer_features(df)

    # Get feature columns
    feature_cols = get_feature_cols(df_eng)

    # Scale features
    X = scaler_X.transform(df_eng[feature_cols])

    # Use last row for prediction (most recent reading)
    X_last = X[[-1]]

    # Current conditions
    last_row = df_eng.iloc[-1]
    current = {
        "timestamp"    : str(last_row['TIMESTAMP']),
        "temperature"  : round(float(last_row['AirTC_Avg']),   2),
        "humidity"     : round(float(last_row['RH_Avg']),      2),
        "pressure"     : round(float(last_row['B_Pressure_Avg']), 2),
        "wind_speed"   : round(float(last_row['WindSpeed_Avg']), 2),
        "rain_mm"      : round(float(last_row['Rain_mm_Tot']),  2),
        "battery_v"    : round(float(last_row['BattV_Avg']),    2),
    }

    # Generate forecasts
    forecasts = []
    for horizon in HORIZONS:
        horizon_forecast = {
            "horizon"    : horizon,
            "confidence" : {},
            "predictions": {},
            "rain_probability": 0.0,
        }

        # Regression predictions
        for target in TARGETS:
            pred = models[target][horizon].predict(X_last)[0]
            pred = float(np.clip(pred, -50, 9999))

            # Apply physical constraints
            if target == 'RH_Avg':
                pred = float(np.clip(pred, 0, 100))
            elif target == 'WindSpeed_Avg':
                pred = float(np.clip(pred, 0, 50))
            elif target == 'B_Pressure_Avg':
                pred = float(np.clip(pred, 800, 900))

            horizon_forecast["predictions"][target] = round(pred, 2)
            horizon_forecast["confidence"][target]  = CONFIDENCE[target][horizon]

        # Rain probability
        rain_prob = float(
            rain_clfs[horizon].predict_proba(X_last)[0][1]
        )
        horizon_forecast["rain_probability"] = round(rain_prob * 100, 1)

        forecasts.append(horizon_forecast)

    return {
        "station"  : "AWS1 Gilgit",
        "location" : {"lat": 35.9, "lon": 74.3, "altitude_m": 1500},
        "generated_at": datetime.utcnow().isoformat(),
        "current"  : current,
        "forecasts": forecasts,
    }

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name"       : "AWS1 Gilgit Weather Forecast API",
        "version"    : "1.0.0",
        "station"    : "AWS1 Gilgit, Pakistan",
        "endpoints"  : {
            "POST /predict" : "Generate forecast from sensor readings",
            "GET  /health"  : "System health check",
            "GET  /docs"    : "Interactive API documentation",
        }
    }

@app.get("/health")
def health():
    return {
        "status"          : "healthy",
        "models_loaded"   : len(TARGETS) * len(HORIZONS),
        "rain_clfs_loaded": len(HORIZONS),
        "targets"         : TARGETS,
        "horizons"        : HORIZONS,
        "timestamp"       : datetime.utcnow().isoformat(),
    }

@app.post("/predict")
def predict(request: ForecastRequest):
    """
    Generate weather forecast from sensor readings.
    Requires at least 288 readings (48 hours of 10-min data).
    """
    if len(request.readings) < 50:
        raise HTTPException(
            status_code = 400,
            detail      = "At least 50 sensor readings required for prediction"
        )

    try:
        df       = readings_to_df(request.readings)
        forecast = generate_forecast(df)
        return forecast

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction error: {str(e)}"
        )
