import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path

cfg = yaml.safe_load(open("configs/default.yaml"))

WEATHER_CSV = cfg.get("raw_weather_csv", "data/raw/weather_2020_2025.csv")
ENERGY_CSV = cfg.get("raw_energy_csv", "data/raw/energy_2023_2024.csv")
HOL_CSV = cfg.get("holidays_csv", "data/AT_public_holidays_2020_2025.csv")
OUT_CSV = cfg.get("engineered_csv", "data/processed/AT_engineered.csv")

# Change these if your new raw CSVs use a different header for dates!
TIME_COL_WEATHER = "Time (UTC)" 
TIME_COL_ENERGY = "Time (UTC)"

Path("data/processed").mkdir(parents=True, exist_ok=True)

def read_csv_try_enc(path, **kwargs):
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

print("Loading and resampling raw datasets...")

# 1. Load Weather Data
weather_df = read_csv_try_enc(WEATHER_CSV, parse_dates=[TIME_COL_WEATHER]).sort_values(TIME_COL_WEATHER)
weather_df[TIME_COL_WEATHER] = pd.to_datetime(weather_df[TIME_COL_WEATHER], utc=True)
weather_df = weather_df.set_index(TIME_COL_WEATHER)
weather_df = weather_df[~weather_df.index.duplicated(keep='first')] # Protect against crashes
weather_df = weather_df.asfreq("15min")

# 2. Load Energy Data
energy_df = read_csv_try_enc(ENERGY_CSV, parse_dates=[TIME_COL_ENERGY]).sort_values(TIME_COL_ENERGY)
energy_df[TIME_COL_ENERGY] = pd.to_datetime(energy_df[TIME_COL_ENERGY], utc=True)
energy_df = energy_df.set_index(TIME_COL_ENERGY)
energy_df = energy_df[~energy_df.index.duplicated(keep='first')] # Protect against crashes
energy_df = energy_df.asfreq("15min")

# 3. Merge on common dates (Inner Join for 2023-2024 intersection)
df = pd.merge(energy_df, weather_df, left_index=True, right_index=True, how="inner")

# Rename columns to standard names
rename_map = {
    "load": "load_mw",
    "temperature_2m_C": "temperature_2m_C",
    "precipitation_mm": "precipitation_mm",
    "mean_global_radiation": "mean_global_radiation",
    "mean_wind_speed": "mean_wind_speed"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# 4. Holidays
if Path(HOL_CSV).exists():
    hol = pd.read_csv(HOL_CSV, parse_dates=["date"])
    hol["date"] = pd.to_datetime(hol["date"], utc=True).dt.floor("D")
    df["is_public_holiday"] = df.index.floor("D").isin(hol["date"]).astype(int)
else:
    df["is_public_holiday"] = 0

df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
df["is_special_day"] = ((df["is_public_holiday"] == 1) | (df["is_weekend"] == 1)).astype(int)

# 5. Calendar cyclical features (Fixed: using fractional hour to preserve names)
idx = df.index
fractional_hour = idx.hour + idx.minute / 60.0  # Maps 15min steps to continuous hour values
df["hour_sin"]  = np.sin(2 * np.pi * fractional_hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * fractional_hour / 24)
df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

# 6. Save
df_out = df.reset_index().rename(columns={"index": "Time (UTC)"})
df_out.to_csv(OUT_CSV, index=False)
print(f"Saved engineered CSV to {OUT_CSV}. Shape: {df.shape}")