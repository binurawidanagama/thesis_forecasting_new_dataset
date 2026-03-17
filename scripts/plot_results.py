# scripts/plot_results.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path("outputs") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = "data/processed/AT_engineered.csv"
TIME_COL = "Time (UTC)"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_full_df():
    """Load the raw CSV for profiling plots."""
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df.set_index(TIME_COL).sort_index()
    return df

def get_series_from_parquet(path: Path, target: str, horizon: int, is_truth: bool = False) -> pd.Series:
    """
    Smart reader: Handles both Baseline (preds.parquet) and dCeNN (raw_energy.parquet) formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # If asking for dCeNN truth, dynamically swap to truth_energy.parquet / truth_weather.parquet
    if is_truth and path.name.startswith("raw_"):
        truth_name = path.name.replace("raw_", "truth_")
        path = path.parent / truth_name

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    # Baseline format (True_target+h / Pred_target+h)
    baseline_col = f"True_{target}+h{horizon}" if is_truth else f"Pred_{target}+h{horizon}"
    if baseline_col in df.columns:
        return df[baseline_col]

    # dCeNN format (target+h)
    dcenn_col = f"{target}+h{horizon}"
    if dcenn_col in df.columns:
        return df[dcenn_col]

    raise KeyError(f"Could not find horizon {horizon} for {target} in {path.name}. Check column names.")


# ---------------------------------------------------------
# 1) Simple profile charts
# ---------------------------------------------------------

def plot_hourly_profile(column="load_mw", out_path=None):
    df = load_full_df()
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")
        
    profile = df.groupby(df.index.hour)[column].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(profile.index, profile.values, marker="o")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(column)
    ax.set_title(f"Average {column} by hour-of-day")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / f"hourly_profile_{column}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_monthly_profile(column="load_mw", out_path=None):
    df = load_full_df()
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")
        
    profile = df.groupby(df.index.month)[column].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(profile.index, profile.values)
    ax.set_xlabel("Month")
    ax.set_xticks(range(1, 13))
    ax.set_ylabel(column)
    ax.set_title(f"Average {column} by month")
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / f"monthly_profile_{column}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------
# 2) Ground truth vs prediction (single model)
# ---------------------------------------------------------

def plot_gt_vs_pred_single(pred_path, target="load_mw", horizon=12, start=None, end=None, model_name="Model", out_path=None):
    s_true = get_series_from_parquet(pred_path, target, horizon, is_truth=True)
    s_pred = get_series_from_parquet(pred_path, target, horizon, is_truth=False)

    if start is not None:
        s_true, s_pred = s_true.loc[start:], s_pred.loc[start:]
    if end is not None:
        s_true, s_pred = s_true.loc[:end], s_pred.loc[:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_true.index, s_true.values, label="Ground truth", linewidth=2)
    ax.plot(s_pred.index, s_pred.values, label=model_name, linestyle="--")
    
    # Calculate step size based on 15-min intervals
    hours_ahead = horizon * 0.25 
    
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(target)
    ax.set_title(f"{target} – Ground truth vs {model_name} ({hours_ahead} hours ahead [h={horizon}])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        safe_target = target.replace("/", "_")
        start_str = (start or "full").replace(":", "-") if isinstance(start, str) else "full"
        out_path = FIG_DIR / f"gt_vs_pred_{safe_target}_h{horizon}_{start_str}.png"

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------
# 3) Ground truth vs prediction (multiple models on one plot)
# ---------------------------------------------------------

def plot_gt_vs_pred_multi(pred_paths, model_names, target="load_mw", horizon=12, start=None, end=None, out_path=None):
    assert len(pred_paths) == len(model_names), "Must provide exactly one name per path."

    # Grab truth from the first path
    s_true = get_series_from_parquet(pred_paths[0], target, horizon, is_truth=True)

    if start is not None:
        s_true = s_true.loc[start:]
    if end is not None:
        s_true = s_true.loc[:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_true.index, s_true.values, label="Ground truth", linewidth=2, color="black", alpha=0.7)

    # Add each model
    for p, name in zip(pred_paths, model_names):
        s_pred = get_series_from_parquet(p, target, horizon, is_truth=False)
        if start is not None:
            s_pred = s_pred.loc[start:]
        if end is not None:
            s_pred = s_pred.loc[:end]
        ax.plot(s_pred.index, s_pred.values, label=name, linestyle="--", alpha=0.8)

    hours_ahead = horizon * 0.25
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(target)
    ax.set_title(f"{target} – Model Comparison ({hours_ahead} hours ahead [h={horizon}])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        safe_target = target.replace("/", "_")
        start_str = (start or "full").replace(":", "-") if isinstance(start, str) else "full"
        out_path = FIG_DIR / f"gt_vs_pred_multi_{safe_target}_h{horizon}_{start_str}.png"

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------
# CLI usage
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis plotting helpers")
    sub = parser.add_subparsers(dest="cmd")

    TARGET_CHOICES = ["load_mw", "temperature_2m_C", "precipitation_mm", "mean_global_radiation", "mean_wind_speed"]

    # hourly profile
    p_hour = sub.add_parser("hourly", help="Average by hour-of-day")
    p_hour.add_argument("--column", default="load_mw", choices=TARGET_CHOICES)

    # monthly profile
    p_month = sub.add_parser("monthly", help="Average by month")
    p_month.add_argument("--column", default="load_mw", choices=TARGET_CHOICES)

    # gt vs pred (single model)
    p_gt1 = sub.add_parser("gt_single", help="Ground truth vs prediction (single model)")
    p_gt1.add_argument("--pred", required=True, help="Path to parquet file (e.g. preds.parquet or raw_energy.parquet)")
    p_gt1.add_argument("--target", default="load_mw", choices=TARGET_CHOICES)
    p_gt1.add_argument("--horizon", type=int, default=12)
    p_gt1.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_gt1.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_gt1.add_argument("--name", default="Model")

    # gt vs pred (multiple models)
    p_gtm = sub.add_parser("gt_multi", help="Ground truth vs multiple models")
    p_gtm.add_argument("--preds", nargs="+", required=True, help="List of prediction parquet paths")
    p_gtm.add_argument("--names", nargs="+", required=True, help="List of model names (same length as preds)")
    p_gtm.add_argument("--target", default="load_mw", choices=TARGET_CHOICES)
    p_gtm.add_argument("--horizon", type=int, default=12)
    p_gtm.add_argument("--start", default=None)
    p_gtm.add_argument("--end", default=None)

    args = parser.parse_args()

    if args.cmd == "hourly":
        plot_hourly_profile(args.column)
    elif args.cmd == "monthly":
        plot_monthly_profile(args.column)
    elif args.cmd == "gt_single":
        plot_gt_vs_pred_single(
            pred_path=args.pred,
            target=args.target,
            horizon=args.horizon,
            start=args.start,
            end=args.end,
            model_name=args.name,
        )
    elif args.cmd == "gt_multi":
        plot_gt_vs_pred_multi(
            pred_paths=args.preds,
            model_names=args.names,
            target=args.target,
            horizon=args.horizon,
            start=args.start,
            end=args.end,
        )
    else:
        parser.print_help()