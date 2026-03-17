import pandas as pd, numpy as np
from pathlib import Path
from .loader import load_engineered

# Treat load as CRITICAL
CRITICAL = ["load_mw"]

# Weather and cyclical features are OPTIONAL
OPTIONAL = [
    "temperature_2m_C", "precipitation_mm", "mean_global_radiation", "mean_wind_speed",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_public_holiday", "is_weekend", "is_special_day"
]

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_numeric(df):
    cols = [c for c in set(CRITICAL+OPTIONAL) if c in df.columns]
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df = _coerce_numeric(df, cols)

    # OPTIONAL: up to 24h interpolation (24 * 4 = 96 steps for 15-min intervals)
    opt = [c for c in OPTIONAL if c in df.columns]
    if opt:
        df[opt] = df[opt].interpolate(limit=96, limit_direction="both")
        df[opt] = df[opt].ffill().bfill()

    # CRITICAL: up to 6h interpolation (6 * 4 = 24 steps for 15-min intervals)
    crit = [c for c in CRITICAL if c in df.columns]
    if crit:
        df[crit] = df[crit].interpolate(limit=24, limit_direction="both").ffill().bfill()

    if crit:
        df = df.dropna(subset=crit)
    return df

def build_master(cfg):
    df = load_engineered(cfg)

    # Add cyclical time features if missing (Using fractional hour to preserve columns)
    idx = df.index.tz_convert("UTC")
    fractional_hour = idx.hour + idx.minute / 60.0
    for k,v in {
        "hour_sin":  np.sin(2*np.pi*fractional_hour/24),
        "hour_cos":  np.cos(2*np.pi*fractional_hour/24),
        "dow_sin":   np.sin(2*np.pi*idx.dayofweek/7),
        "dow_cos":   np.cos(2*np.pi*idx.dayofweek/7),
        "month_sin": np.sin(2*np.pi*(idx.month-1)/12),
        "month_cos": np.cos(2*np.pi*(idx.month-1)/12),
    }.items():
        if k not in df.columns: df[k] = v

    # clean before split
    df = _clean_numeric(df)

    # splits
    split = cfg["time"]["split"]
    train_end = pd.Timestamp(split["train_until"], tz="UTC")
    val_end   = pd.Timestamp(split["val_until"],   tz="UTC")
    test_end  = pd.Timestamp(split["test_until"],  tz="UTC")

    df = df.loc[:test_end]
    train_df = df.loc[:train_end]
    val_df   = df.loc[train_end + pd.Timedelta(minutes=15): val_end]
    test_df  = df.loc[val_end + pd.Timedelta(minutes=15):]

    outdir = Path(cfg["paths"]["interim_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(outdir/"train.parquet")
    val_df.to_parquet(outdir/"val.parquet")
    test_df.to_parquet(outdir/"test.parquet")
    return train_df, val_df, test_df