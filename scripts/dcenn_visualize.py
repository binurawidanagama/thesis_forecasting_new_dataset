"""
Thesis Visualizer: 15-Minute Interval Version
---------------------------------------------
LOGIC:
1. Data frequency is 15 minutes.
2. Lookbacks are converted from hourly to 15-min steps:
      24h  ->  96
      72h  -> 288
      168h -> 672
3. Horizons remain [12, 24, 72] as 15-minute steps.
4. ALL forecasts (dCeNN, CNN, LSTM) and the Ground Truth are shifted 
   forward by the current horizon step (`hs`) to align Forecast Origin 
   with the actual Target Time on the plots.
5. Safe filename sanitization is preserved.

Usage:
  python scripts/dcenn_visualize.py --compare_all --start_date 2024-11-01 --length_steps 672
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# CONFIGURATION
# -----------------------------
INTERVAL_MINUTES = 15

# Converted lookbacks for 15-minute data
LOOKBACKS = [96, 288, 672]

# Horizons remain as 15-minute steps
HORIZONS = [12, 24, 72]

PREFERRED_TARGETS: Dict[str, List[str]] = {
    "energy": ["load_mw"],
    "weather": [
        "temperature_2m_C",
        "mean_global_radiation",
        "precipitation_mm",
        "mean_wind_speed",
    ],
}

# -----------------------------
# DATA UTILITIES
# -----------------------------
def ensure_timestamp(df: pd.DataFrame, name: str = "DF") -> pd.DataFrame:
    d = df.copy()

    if "timestamp" in d.columns:
        ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(d.index, utc=True, errors="coerce")

    d["timestamp"] = ts
    d = d.set_index("timestamp", drop=False)
    d.index.name = None
    d = d.dropna(subset=["timestamp"]).sort_index()
    return d


def time_slice(
    df: pd.DataFrame,
    start_date: Optional[str],
    length_steps: int
) -> pd.DataFrame:
    """
    Slice by number of 15-minute steps, not hours.
    """
    d = ensure_timestamp(df)
    if len(d) == 0:
        return d

    if start_date:
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        end_ts = start_ts + pd.Timedelta(minutes=INTERVAL_MINUTES * length_steps)
        sub = d.loc[(d.index >= start_ts) & (d.index < end_ts)]

        if len(sub) == 0:
            return d.iloc[-min(length_steps, len(d)):]
        return sub

    n = len(d)
    if n <= length_steps:
        return d

    start_idx = min(max(n // 2, 0), max(n - length_steps, 0))
    return d.iloc[start_idx:start_idx + length_steps]


def align_dfs(dfs_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    valid_dfs = {k: v for k, v in dfs_dict.items() if v is not None and not v.empty}
    if not valid_dfs:
        return {}

    common_idx = None
    for df in valid_dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        return {}

    return {k: df.reindex(common_idx) for k, df in valid_dfs.items()}


# -----------------------------
# COLUMN FINDER & SHIFTER
# -----------------------------
def get_col_name(df: pd.DataFrame, feat: str, h_step: int, prefix: str = "") -> Optional[str]:
    candidates = [
        f"{prefix}{feat}+h{h_step}",
        f"{feat}+h{h_step}",
        f"{prefix}{feat}",
        feat,
    ]
    cols = list(df.columns)

    for cand in candidates:
        if cand in cols:
            return cand
        for c in cols:
            if cand in c:
                return c
    return None


def extract_and_shift(
    df: pd.DataFrame,
    col_name: str,
    shift_steps: int,
) -> Optional[pd.Series]:
    """
    Extract the column and shift timestamp by shift_steps * 15 minutes.
    """
    if col_name not in df.columns:
        return None

    series = df[col_name].copy()

    if shift_steps != 0:
        series.index = series.index + pd.Timedelta(minutes=shift_steps * INTERVAL_MINUTES)

    return series


# -----------------------------
# PLOTTING
# -----------------------------
def plot_models(data_map: Dict[str, pd.Series], feat: str, title: str, save_path: Path):
    if not data_map:
        return

    plt.figure(figsize=(12, 5))

    order = ["Truth", "CNN", "LSTM", "dCeNN Raw", "dCeNN + ASP"]
    order += [k for k in data_map.keys() if k not in order]

    styles = {
        "Truth":       {"color": "black",      "alpha": 0.5, "lw": 2.5, "ls": "-"},
        "dCeNN Raw":   {"color": "tab:blue",   "alpha": 0.8, "lw": 1.5, "ls": "--"},
        "dCeNN + ASP": {"color": "tab:green",  "alpha": 1.0, "lw": 2.0, "ls": "-"},
        "CNN":         {"color": "tab:red",    "alpha": 0.8, "lw": 1.5, "ls": "-"},
        "LSTM":        {"color": "tab:orange", "alpha": 0.8, "lw": 1.5, "ls": "-"},
    }

    first_key = next(iter(data_map))
    timestamps = data_map[first_key].index

    for label in order:
        if label in data_map:
            series = data_map[label]
            st = styles.get(label, {"alpha": 0.8, "lw": 1.5})
            plt.plot(timestamps, series.values, label=label, **st)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=0)

    plt.title(title, fontsize=14)
    plt.ylabel(feat, fontsize=12)
    plt.xlabel("Time (UTC) [Month-Day]", fontsize=10)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# MAIN LOGIC
# -----------------------------
def load_data_wrapper(
    out_dir: Path,
    energy_root: Path,
    weather_root: Path,
    cnn_root: Path,
    lstm_root: Path,
    start_date: Optional[str],
    length_steps: int,
    h_steps_arg: str,
    mode: str,
):
    print(f"\n[INFO] Starting Visualization... Mode: {mode}")
    print(f"[INFO] Using {INTERVAL_MINUTES}-minute interval data")
    print(f"[INFO] Lookbacks: {LOOKBACKS}")
    print(f"[INFO] Horizons: {HORIZONS}")

    for lb in LOOKBACKS:
        for hz in HORIZONS:

            if h_steps_arg == "max":
                steps = [hz]
            elif h_steps_arg == "auto":
                steps = sorted(list(set([1, max(1, hz // 2), hz])))
            else:
                steps = [int(x) for x in h_steps_arg.split(",")]

            flatten = (len(steps) == 1)

            for task in ["energy", "weather"]:
                dcenn_root = energy_root if task == "energy" else weather_root

                path_raw = dcenn_root / f"LB{lb}_H{hz}" / f"raw_{task}.parquet"
                path_true = dcenn_root / f"LB{lb}_H{hz}" / f"truth_{task}.parquet"
                path_clean = dcenn_root / f"LB{lb}_H{hz}" / f"clean_{task}.parquet"

                if not path_raw.exists():
                    continue

                df_raw = ensure_timestamp(pd.read_parquet(path_raw), "dCeNN Raw")
                df_true = ensure_timestamp(pd.read_parquet(path_true), "Truth")
                df_clean = ensure_timestamp(pd.read_parquet(path_clean), "dCeNN + ASP") if path_clean.exists() else None

                df_cnn, df_lstm = None, None
                if mode == "compare_all":
                    p_cnn = cnn_root / f"LB{lb}_H{hz}_{task.upper()}" / "preds.parquet"
                    p_lstm = lstm_root / f"LB{lb}_H{hz}_{task.upper()}" / "preds.parquet"

                    if p_cnn.exists():
                        df_cnn = ensure_timestamp(pd.read_parquet(p_cnn), "CNN")
                    if p_lstm.exists():
                        df_lstm = ensure_timestamp(pd.read_parquet(p_lstm), "LSTM")

                for hs in steps:
                    base_out = out_dir / ("COMPARE_ALL" if mode == "compare_all" else "DCENN_ONLY")
                    folder = base_out / f"LB{lb}_H{hz}" / task
                    if not flatten:
                        folder = folder / f"h{hs}"

                    valid_feats = [c.split("+")[0] for c in df_raw.columns if f"+h{hs}" in c]
                    valid_feats = list(set(valid_feats).intersection(PREFERRED_TARGETS[task]))

                    for feat in valid_feats:
                        col_dcenn = get_col_name(df_raw, feat, hs)
                        col_true = get_col_name(df_true, feat, hs)

                        if not col_dcenn or not col_true:
                            continue

                        plot_data: Dict[str, Optional[pd.Series]] = {}

                        # 1. Truth and ALL models shifted by the current step `hs`
                        plot_data["Truth"] = extract_and_shift(
                            df_true, col_true, shift_steps=hs
                        )
                        plot_data["dCeNN Raw"] = extract_and_shift(
                            df_raw, col_dcenn, shift_steps=hs
                        )

                        if df_clean is not None:
                            col_clean = get_col_name(df_clean, feat, hs)
                            if col_clean:
                                plot_data["dCeNN + ASP"] = extract_and_shift(
                                    df_clean, col_clean, shift_steps=hs
                                )

                        # 2. Baselines are shifted too (Aligns perfectly with Ground Truth)
                        if df_cnn is not None:
                            col_cnn = get_col_name(df_cnn, feat, hs, "Pred_")
                            if col_cnn:
                                plot_data["CNN"] = extract_and_shift(
                                    df_cnn, col_cnn, shift_steps=hs
                                )

                        if df_lstm is not None:
                            col_lstm = get_col_name(df_lstm, feat, hs, "Pred_")
                            if col_lstm:
                                plot_data["LSTM"] = extract_and_shift(
                                    df_lstm, col_lstm, shift_steps=hs
                                )

                        # Align and slice
                        df_map = {k: v.to_frame(name=k) for k, v in plot_data.items() if v is not None}
                        df_map = {k: time_slice(v, start_date, length_steps) for k, v in df_map.items()}
                        aligned_map = align_dfs(df_map)
                        final_series = {k: v.iloc[:, 0] for k, v in aligned_map.items()}

                        if "Truth" not in final_series or len(final_series["Truth"]) == 0:
                            continue

                        safe_feat = (
                            feat.replace(" ", "_")
                                .replace("(", "")
                                .replace(")", "")
                                .replace("/", "-")
                        )

                        fname = f"{task}_{safe_feat}_h{hs}.png"
                        title = f"{task.upper()} | {feat} | LB{lb} H{hz} (step +{hs}, 15min)"
                        plot_models(final_series, feat, title, folder / fname)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="thesis_plots_final_15min")
    ap.add_argument("--dcenn_energy_root", type=str, default="outputs_energy_full")
    ap.add_argument("--dcenn_weather_root", type=str, default="outputs_weather_full")
    ap.add_argument("--cnn_root", type=str, default="artifacts_cnn_baseline")
    ap.add_argument("--lstm_root", type=str, default="artifacts_lstm_baseline")

    ap.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument(
        "--length_steps",
        type=int,
        default=672,
        help="Number of 15-minute steps to plot (e.g. 672 = 7 days)"
    )
    ap.add_argument("--h_steps", type=str, default="max")
    ap.add_argument("--compare_all", action="store_true")

    args = ap.parse_args()
    mode = "compare_all" if args.compare_all else "dcenn_only"

    load_data_wrapper(
        Path(args.out_dir),
        Path(args.dcenn_energy_root),
        Path(args.dcenn_weather_root),
        Path(args.cnn_root),
        Path(args.lstm_root),
        args.start_date,
        args.length_steps,
        args.h_steps,
        mode,
    )

    print(f"\n[DONE] Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()