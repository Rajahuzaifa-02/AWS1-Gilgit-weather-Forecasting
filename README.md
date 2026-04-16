---
title: AWS1 Gilgit Weather Forecast
emoji: 🌤
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
---

# AWS1 Gilgit Weather Forecast System

A machine learning based weather forecasting system for Gilgit, Pakistan.

## About
- **Station:** AWS1 Gilgit (Campbell Scientific CR350)
- **Location:** Gilgit, Pakistan (35.9°N, 74.3°E, ~1500m)
- **Model:** XGBoost ensemble (25 models)
- **Training period:** May 2024 – August 2025
- **Forecast horizons:** +1hr, +3hr, +6hr, +12hr, +24hr

## Variables Forecast
| Variable | Best R² | Reliability |
|----------|---------|-------------|
| Temperature | 0.991 | ✅ High |
| Humidity | 0.964 | ✅ High (short term) |
| Pressure | 0.995 | ✅ High |
| Wind Speed | 0.559 | ⚠️ +1hr only |
| Rain | F1=0.524 | ⚠️ Probability only |

## How to Use
1. Upload your cleaned sensor CSV file, or use sample data
2. View current conditions and 24-hour forecast
3. Check confidence levels for each variable

## Technical Details
- Feature engineering with 86 weather-aware features
- Physical meteorological relationships encoded
- Pressure extrapolation for improved accuracy
- Wind climatology features for diurnal patterns
- Safe lag features (≥24hr) to prevent data leakage
