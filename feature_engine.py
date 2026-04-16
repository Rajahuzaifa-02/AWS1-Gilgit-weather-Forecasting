"""
Feature Engineering Pipeline
==============================
Transforms raw sensor readings into model-ready features.
Must match training pipeline exactly.
"""

import numpy as np
import pandas as pd


# ── Constants from training data ─────────────────────────────────────────────
HOURLY_WIND_MEAN = {
    0:0.686, 1:0.658, 2:0.605, 3:0.598, 4:0.538,
    5:0.514, 6:0.499, 7:0.571, 8:0.764, 9:0.960,
    10:1.060, 11:1.129, 12:1.262, 13:1.329, 14:1.383,
    15:1.461, 16:1.417, 17:1.347, 18:1.251, 19:1.105,
    20:0.964, 21:0.872, 22:0.763, 23:0.690
}

MONTHLY_WIND_MEAN = {
    1:0.611, 2:0.822, 3:1.143, 4:1.037, 5:1.180,
    6:1.345, 7:1.267, 8:1.109, 9:0.930,
    10:0.704, 11:0.515, 12:0.466
}

MONTHLY_TEMP_MEAN = {
    1:4.33,  2:7.20,  3:13.50, 4:18.90, 5:23.80,
    6:28.10, 7:30.20, 8:29.50, 9:24.60,
    10:17.80, 11:9.50, 12:4.80
}

TRAIN_MONTHLY_PRESSURE = {
    1:853.507, 2:849.980, 3:850.010,
    4:849.140, 5:843.810, 6:841.230,
    7:837.000, 8:839.380, 9:841.660,
    10:843.980, 11:844.930, 12:845.590,
}

HORIZON = 144  # 24 hours in 10-min steps


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline to dataframe.
    Input dataframe must have TIMESTAMP and sensor columns.
    Returns dataframe with all engineered features.
    """
    df = df.copy()
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    T  = df['AirTC_Avg']
    RH = df['RH_Avg'].clip(lower=0.1)
    P  = df['B_Pressure_Avg']
    W  = df['WindSpeed_Avg']

    hour  = df['TIMESTAMP'].dt.hour + df['TIMESTAMP'].dt.minute / 60
    doy   = df['TIMESTAMP'].dt.dayofyear
    month = df['TIMESTAMP'].dt.month
    hour_int = df['TIMESTAMP'].dt.hour

    # ── Group 1: Cyclic time ──────────────────────────────────────────────────
    df['hour_sin']  = np.sin(2 * np.pi * hour  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hour  / 24)
    df['doy_sin']   = np.sin(2 * np.pi * doy   / 365)
    df['doy_cos']   = np.cos(2 * np.pi * doy   / 365)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['is_daytime']= ((df['TIMESTAMP'].dt.hour >= 6) &
                       (df['TIMESTAMP'].dt.hour <= 18)).astype(int)

    # ── Group 2: Physical interactions ───────────────────────────────────────
    df['feat_temp_rh_product']     = T * RH
    df['feat_temp_rh_ratio']       = T / (RH + 1)
    df['feat_temp_pressure_ratio'] = T / P
    df['feat_pressure_wind']       = W / (P - 820).clip(lower=0.1)
    df['feat_temp_wind_product']   = T * W
    df['feat_rh_above_90']         = (RH > 90).astype(int)
    df['feat_rh_above_80']         = (RH > 80).astype(int)
    df['feat_rh_above_70']         = (RH > 70).astype(int)

    # ── Group 3: Meteorological quantities ───────────────────────────────────
    df['feat_dew_point']            = T - ((100 - RH) / 5.0)
    df['feat_dew_point_depression'] = T - df['feat_dew_point']
    df['feat_temp_range']           = df['AirTC_Max'] - df['AirTC_Min']
    df['feat_rh_range']             = df['RH_Max']    - df['RH_Min']
    df['feat_wind_gust_range']      = df['WindSpeed_Max'] - df['WindSpeed_Min']

    # ── Group 4: Pressure tendency ───────────────────────────────────────────
    df['feat_pressure_tendency_1h'] = P - P.shift(6)
    df['feat_pressure_tendency_3h'] = P - P.shift(18)
    df['feat_pressure_tendency_6h'] = P - P.shift(36)
    df['feat_pressure_falling']     = (
        df['feat_pressure_tendency_3h'] < -1.0).astype(int)
    df['feat_pressure_rising']      = (
        df['feat_pressure_tendency_3h'] >  1.0).astype(int)
    df['feat_pressure_accel']       = (
        df['feat_pressure_tendency_3h'] -
        df['feat_pressure_tendency_3h'].shift(18))
    df['feat_rh_trend_3h']          = RH - RH.shift(18)
    df['feat_temp_trend_3h']        = T  - T.shift(18)

    # ── Group 5: Solar proxy ─────────────────────────────────────────────────
    df['feat_solar_proxy']        = df['PTemp_C_Avg']
    df['feat_solar_heating_rate'] = df['PTemp_C_Avg'] - df['PTemp_C_Avg'].shift(6)

    # ── Group 6: Wind direction ──────────────────────────────────────────────
    df['feat_winddir_sin']       = np.sin(np.deg2rad(df['WindDir_D1_WVT']))
    df['feat_winddir_cos']       = np.cos(np.deg2rad(df['WindDir_D1_WVT']))
    df['feat_winddir_change_1h'] = df['WindDir_D1_WVT'] - df['WindDir_D1_WVT'].shift(6)

    # ── Group 7: Rain context ────────────────────────────────────────────────
    df['feat_rain_6h_sum']  = df['Rain_mm_Tot'].rolling(36,  min_periods=1).sum()
    df['feat_rain_24h_sum'] = df['Rain_mm_Tot'].rolling(144, min_periods=1).sum()
    df['feat_rain_event']   = (df['Rain_mm_Tot'] > 0).astype(int)
    df['feat_low_battery']  = (df['BattV_Avg'] < 11.5).astype(int)

    # ── Group 8: Safe lag features ───────────────────────────────────────────
    lag_map = {
        'temp'    : 'AirTC_Avg',
        'rh'      : 'RH_Avg',
        'pressure': 'B_Pressure_Avg',
        'wind'    : 'WindSpeed_Avg',
        'rain'    : 'Rain_mm_Tot',
    }
    for short, col in lag_map.items():
        df[f'lag_{short}_24h'] = df[col].shift(HORIZON)
        df[f'lag_{short}_48h'] = df[col].shift(HORIZON * 2)

    # ── Group 9: Rolling statistics ──────────────────────────────────────────
    roll_map = {
        'temp'    : 'AirTC_Avg',
        'rh'      : 'RH_Avg',
        'pressure': 'B_Pressure_Avg',
        'wind'    : 'WindSpeed_Avg',
    }
    for short, col in roll_map.items():
        shifted = df[col].shift(HORIZON)
        df[f'roll_{short}_mean_24h'] = shifted.rolling(HORIZON, min_periods=1).mean()
        df[f'roll_{short}_std_24h']  = shifted.rolling(HORIZON, min_periods=1).std().fillna(0)

    # ── Stage 3b: Pressure seasonal ──────────────────────────────────────────
    df['feat_pressure_seasonal_mean'] = month.map(TRAIN_MONTHLY_PRESSURE)
    df['feat_pressure_anomaly']       = P - df['feat_pressure_seasonal_mean']
    df['feat_winddir_persistence']    = (
        np.abs(df['feat_winddir_sin'] - df['feat_winddir_sin'].shift(144)) +
        np.abs(df['feat_winddir_cos'] - df['feat_winddir_cos'].shift(144))
    ).fillna(0)

    # ── Stage 3c: Pressure extrapolation ─────────────────────────────────────
    tend_1h  = df['feat_pressure_tendency_1h']
    tend_3h  = df['feat_pressure_tendency_3h'] / 3
    tend_avg = (tend_1h + tend_3h) / 2

    df['feat_pressure_extrap_1hr']  = P + (tend_avg * 1)
    df['feat_pressure_extrap_3hr']  = P + (tend_avg * 3)
    df['feat_pressure_extrap_6hr']  = P + (tend_avg * 6)
    df['feat_pressure_extrap_12hr'] = P + (tend_avg * 12)
    df['feat_pressure_extrap_24hr'] = P + (tend_avg * 24)
    df['feat_current_pressure']     = P

    # ── Stage 3d: Wind features ──────────────────────────────────────────────
    df['feat_wind_tendency_1h']  = W - W.shift(6)
    df['feat_wind_tendency_3h']  = W - W.shift(18)
    df['feat_wind_accel']        = (
        df['feat_wind_tendency_1h'] -
        df['feat_wind_tendency_1h'].shift(6)
    )
    df['feat_wind_sustained_3h'] = (W > 1.0).rolling(18, min_periods=1).sum()
    df['feat_wind_sustained_6h'] = (W > 1.0).rolling(36, min_periods=1).sum()
    df['feat_wind_calm_flag']    = (W < 0.3).astype(int)
    df['feat_wind_gusty_flag']   = (df['feat_wind_gust_range'] > 2.0).astype(int)

    wind_dir_rad = np.deg2rad(df['WindDir_D1_WVT'])
    df['feat_wind_u']          = W * np.sin(wind_dir_rad)
    df['feat_wind_v']          = W * np.cos(wind_dir_rad)
    df['feat_wind_u_lag24h']   = df['feat_wind_u'].shift(144)
    df['feat_wind_v_lag24h']   = df['feat_wind_v'].shift(144)
    df['feat_wind_u_tendency'] = df['feat_wind_u'] - df['feat_wind_u'].shift(6)
    df['feat_wind_v_tendency'] = df['feat_wind_v'] - df['feat_wind_v'].shift(6)

    df['feat_cumulative_heating_6h'] = (
        df['feat_solar_heating_rate'].clip(lower=0)
        .rolling(36, min_periods=1).sum()
    )
    df['feat_cumulative_heating_3h'] = (
        df['feat_solar_heating_rate'].clip(lower=0)
        .rolling(18, min_periods=1).sum()
    )

    w_tend_avg = (df['feat_wind_tendency_1h'] +
                  df['feat_wind_tendency_3h'] / 3) / 2
    df['feat_wind_extrap_1hr']  = (W + w_tend_avg * 1).clip(lower=0)
    df['feat_wind_extrap_3hr']  = (W + w_tend_avg * 3).clip(lower=0)
    df['feat_wind_extrap_6hr']  = (W + w_tend_avg * 6).clip(lower=0)
    df['feat_wind_extrap_12hr'] = (W + w_tend_avg * 12).clip(lower=0)
    df['feat_wind_extrap_24hr'] = (W + w_tend_avg * 24).clip(lower=0)
    df['feat_current_wind']     = W

    # Fill NaN
    df = df.ffill().bfill()

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature columns in correct order for model input."""
    ALL_TARGETS = [
        'AirTC_Avg', 'RH_Avg', 'B_Pressure_Avg',
        'WindSpeed_Avg', 'Rain_mm_Tot'
    ]
    EXCLUDE = ['TIMESTAMP', 'year', 'month'] + ALL_TARGETS + [
        'AirTC_Min',     'AirTC_Max',
        'RH_Min',        'RH_Max',
        'WindSpeed_Min', 'WindSpeed_Max',
        'WindSpeed_S_WVT',
        'WindDir_D1_WVT','WindDir_SD1_WVT',
        'BattV_Avg',
    ]
    return [c for c in df.columns if c not in EXCLUDE]
