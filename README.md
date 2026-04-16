# 🌤 AWS1 Gilgit Weather Forecast System

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Huzii48662/aws1-gilgit-weather-forecast)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)

A production-deployed machine learning weather forecasting system
for the AWS1 Automatic Weather Station in Gilgit, Pakistan.

## 🚀 Live Demo
**[→ Open Dashboard](https://huggingface.co/spaces/Huzii48662/aws1-gilgit-weather-forecast)**

## 📍 Station Details
| Parameter | Value |
|-----------|-------|
| Station   | AWS1 Gilgit |
| Logger    | Campbell Scientific CR350 |
| Location  | 35.9°N, 74.3°E |
| Altitude  | ~1500m |
| Interval  | 10-minute |
| Data      | May 2024 – Feb 2026 |

## 🎯 What It Does
Forecasts 5 weather variables at 5 time horizons:

| Variable | +1hr | +3hr | +6hr | +12hr | +24hr |
|----------|------|------|------|-------|-------|
| Temperature (°C) | R²=0.991 | R²=0.978 | R²=0.943 | R²=0.900 | R²=0.872 |
| Humidity (%) | R²=0.964 | R²=0.907 | R²=0.821 | R²=0.663 | R²=0.455 |
| Pressure (hPa) | R²=0.995 | R²=0.983 | R²=0.940 | R²=0.825 | R²=0.769 |
| Wind Speed (m/s) | R²=0.559 | R²=0.330 | R²=0.212 | R²=0.001 | R²=0.163 |
| Rain (F1 Score) | 0.509 | 0.462 | 0.479 | 0.497 | 0.524 |

## 🏗️ Architecture---
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
