import pandas as pd
from pathlib import Path

def load_engineered(cfg):
    p = Path(cfg["paths"]["engineered_csv"])
    c = cfg["columns"]
    ts = c["timestamp"]
    
    # Read the CSV
    df = pd.read_csv(p, parse_dates=[ts])
    
    # Force UTC timezone
    df[ts] = pd.to_datetime(df[ts], utc=True)
    
    # Set index and force 15-minute frequency!
    df = df.set_index(ts).sort_index().asfreq("15min")

    # (We removed the old hardcoded column renaming block here because 
    # our preprocess script already names everything perfectly!)

    # Unify holiday flags into a single 'holiday' column safely
    hol_keys = cfg["columns"].get("holiday_flags", [])
    if hol_keys:
        available_hol_keys = []
        for h in hol_keys:
            if h in df.columns:
                df[h] = (df[h].astype(int) > 0).astype(int)
                available_hol_keys.append(h)
                
        # Only sum columns that actually exist in the dataframe
        if available_hol_keys:
            df["holiday"] = (df[available_hol_keys].sum(axis=1) > 0).astype(int)
        else:
            df["holiday"] = 0
    else:
        df["holiday"] = 0

    return df