"""
Universal Visualizer for split Tasks.
Plots both ENERGY and WEATHER forecasts.
Saves plots to a clean separate folder structure.

Usage:
  python scripts/cnn_lstm_visualize.py --model cnn --out_dir plots --lookback 96 --horizon 12
  python scripts/cnn_lstm_visualize.py --model lstm --out_dir thesis_images --lookback 288 --horizon 24
  python scripts/cnn_lstm_visualize.py --model cnn (plots EVERYTHING to default 'plots' folder)
  python scripts/cnn_lstm_visualize.py --model cnn --horizon 72 (plots all lookbacks for H72)

  # Comparison Plots (CNN vs LSTM)
  python scripts/cnn_lstm_visualize.py --compare
  python scripts/cnn_lstm_visualize.py --compare --horizon 72

  # Plot specific week (e.g. Winter vs Summer)
  python scripts/cnn_lstm_visualize.py --compare --start_date 2024-01-10
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import glob
import os

# --- CONFIG: Features to Visualize ---
# MATCHES your new 15-minute dataset targets
TARGETS_TO_PLOT = [
    "load_mw", 
    "temperature_2m_C", 
    "mean_global_radiation",
    "precipitation_mm", 
    "mean_wind_speed"
]

def get_subset(df, start_date=None, length_hours=168):
    """
    Slices DataFrame by date if provided, otherwise uses default index.
    Default length 168 = 1 Week (672 steps at 15-min intervals).
    """
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    if start_date:
        # Date-based slicing (Best for Thesis)
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
            
        end_ts = start_ts + pd.Timedelta(hours=length_hours)
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
        subset = df.loc[mask]
        
        if len(subset) == 0:
            print(f"[WARN] No data found for {start_date}. Falling back to default.")
            return df.iloc[-length_hours * 4:] # 4 steps per hour
        return subset
    else:
        # Index-based slicing (Fallback)
        # Default to a slice in the middle of the dataset
        start_idx = 4000 if len(df) > 4200 else 0
        return df.iloc[start_idx : start_idx + (length_hours * 4)]

def plot_single(df, feature_name, title, save_path, start_date, model_color="#d62728"):
    """Standard single-model plot."""
    true_col = f"True_{feature_name}"
    pred_col = f"Pred_{feature_name}"
    
    # Handle matching logic 
    found_true = [c for c in df.columns if true_col in c]
    found_pred = [c for c in df.columns if pred_col in c]
    
    if not found_true or not found_pred: return
    
    # Use the specific column name found
    t_col = found_true[0]
    p_col = found_pred[0]

    subset = get_subset(df, start_date)
    
    plt.figure(figsize=(12, 5))
    plt.plot(subset["timestamp"], subset[t_col], label="Actual", color="black", linewidth=1.5, alpha=0.6)
    plt.plot(subset["timestamp"], subset[p_col], label="Forecast", color=model_color, linewidth=1.5, alpha=0.9)
    
    # Format x-axis nicely for the thesis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=0)
    
    plt.title(title, fontsize=14)
    plt.ylabel(feature_name, fontsize=12)
    plt.xlabel("Time (UTC) [Month-Day]", fontsize=10)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison(df_cnn, df_lstm, feature_name, title, save_path, start_date):
    """Overlays CNN (Red) and LSTM (Blue) vs Actual (Black)."""
    true_col_stub = f"True_{feature_name}"
    pred_col_stub = f"Pred_{feature_name}"
    
    # Fuzzy match columns
    cnn_cols = df_cnn.columns
    t_col = next((c for c in cnn_cols if true_col_stub in c), None)
    p_col = next((c for c in cnn_cols if pred_col_stub in c), None)
    
    if not t_col or not p_col: return

    # Get Subsets
    sub_cnn = get_subset(df_cnn, start_date)
    sub_lstm = get_subset(df_lstm, start_date)
    
    # Align timestamps (Use CNN as master time)
    timestamps = sub_cnn["timestamp"]
    
    plt.figure(figsize=(12, 5))
    
    # 1. Actual (Black)
    plt.plot(timestamps, sub_cnn[t_col], label="Actual", color="black", linewidth=2.0, alpha=0.4)
    
    # 2. LSTM (Blue - Dashed)
    # Ensure lengths match before plotting
    if len(sub_lstm) == len(timestamps):
        plt.plot(timestamps, sub_lstm[p_col], label="LSTM", color="tab:blue", linewidth=1.5, linestyle="--", alpha=0.9)

    # 3. CNN (Red - Solid)
    plt.plot(timestamps, sub_cnn[p_col], label="CNN", color="tab:red", linewidth=1.5, alpha=0.9)
    
    # Format x-axis nicely for the thesis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=0)
    
    plt.title(title, fontsize=14)
    plt.ylabel(feature_name, fontsize=12)
    plt.xlabel("Time (UTC) [Month-Day]", fontsize=10)
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"   [COMPARE] Saved {save_path}")
    plt.close()

def run_visualization(model_type, compare_mode, target_lb, target_hz, out_root, start_date):
    # ---------------------------
    # MODE A: Individual Plots
    # ---------------------------
    if not compare_mode:
        src_dir = f"artifacts_{model_type}_baseline"
        if not os.path.exists(src_dir):
            print(f"[ERROR] {src_dir} not found.")
            return

        pred_files = glob.glob(os.path.join(src_dir, "*", "preds.parquet"))
        for f in pred_files:
            folder_name = os.path.basename(os.path.dirname(f))
            
            if target_lb and f"LB{target_lb}" not in folder_name: continue
            if target_hz and f"H{target_hz}" not in folder_name: continue

            print(f"Processing {folder_name}...")
            try: 
                df = pd.read_parquet(f)
            except Exception as e: 
                print(f"Skipping {folder_name}: {e}")
                continue
            
            dest = os.path.join(out_root, model_type.upper(), folder_name)
            os.makedirs(dest, exist_ok=True)
            
            for target in TARGETS_TO_PLOT:
                plot_single(df, target, f"{model_type.upper()} ({folder_name})", 
                           os.path.join(dest, f"{model_type}_{target}.png"), start_date)

    # ---------------------------
    # MODE B: Comparison Mode (CNN vs LSTM)
    # ---------------------------
    else:
        print("\n--- RUNNING COMPARISON MODE (CNN vs LSTM) ---")
        cnn_dir = "artifacts_cnn_baseline"
        lstm_dir = "artifacts_lstm_baseline"
        
        if not os.path.exists(cnn_dir) or not os.path.exists(lstm_dir):
            print(f"[ERROR] Cannot compare: Missing Baseline artifact folders.")
            return

        cnn_folders = [os.path.basename(d) for d in glob.glob(os.path.join(cnn_dir, "*"))]
        
        for folder_name in cnn_folders:
            if target_lb and f"LB{target_lb}" not in folder_name: continue
            if target_hz and f"H{target_hz}" not in folder_name: continue
            
            path_c = os.path.join(cnn_dir, folder_name, "preds.parquet")
            path_l = os.path.join(lstm_dir, folder_name, "preds.parquet")
            
            if not os.path.exists(path_l) or not os.path.exists(path_c): 
                continue

            print(f"Comparing {folder_name}...")
            try:
                df_c = pd.read_parquet(path_c)
                df_l = pd.read_parquet(path_l)
            except Exception as e: 
                print(f"Skipping {folder_name}: {e}")
                continue
            
            dest = os.path.join(out_root, "COMPARISON", folder_name)
            os.makedirs(dest, exist_ok=True)
            
            for target in TARGETS_TO_PLOT:
                # Replace tricky characters in the filename so it doesn't create subfolders
                safe_target = target.replace("/", "-").replace("(", "").replace(")", "").replace(" ", "_")
                out_name = f"COMPARE_{safe_target}.png"
                title = f"CNN vs LSTM ({folder_name})"
                plot_comparison(df_c, df_l, target, title, os.path.join(dest, out_name), start_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cnn", "lstm"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    
    # NEW: Date selection for thesis figures
    parser.add_argument("--start_date", type=str, default=None, help="e.g. 2024-11-01")
    
    args = parser.parse_args()
    
    if not args.model and not args.compare:
        print("Error: Specify --model [cnn/lstm] OR --compare")
    else:
        run_visualization(args.model, args.compare, args.lookback, args.horizon, args.out_dir, args.start_date)