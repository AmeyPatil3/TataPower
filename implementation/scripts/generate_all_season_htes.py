"""
Row-Level HTE Generation Engine for All Seasons.
Replicates the exact CausalForestDML methodology from:
  models/2024 models/with socio/sociowinter(2)_row_level_HTE.ipynb

Fits:
1. Weather CausalForestDML:
   Y = wdLoad, T = [temp, rh], X = [hour, weekend, holiday, festival, office_hours]
   Yields tuple-level temp_hte and rh_hte.
2. Calendar CausalForestDML:
   Y = wdLoad, T = [weekend, holiday, festival, office_hours], X = [temp, rh, hour]
   Yields tuple-level weekend_hte, holiday_hte, festival_hte, office_hte.

Saves row-level HTE datasets to:
  - models/2024 models/with socio/<season>_2024_with_hte.csv
  - models/2024 models/with socio/<season>_2024_with_all_hte.csv
  - data/2024 data/with socio/<season>_2024_with_hte.csv
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

INPUT_SEASONS = {
    "winter": os.path.join(WORKSPACE_ROOT, "data", "2024 data", "with socio", "winter_2024.csv"),
    "summer": os.path.join(WORKSPACE_ROOT, "data", "2024 data", "with socio", "summer_2024.csv"),
    "monsoon": os.path.join(WORKSPACE_ROOT, "data", "2024 data", "with socio", "monsoon_2024.csv"),
    "autumn": os.path.join(WORKSPACE_ROOT, "data", "2024 data", "with socio", "autumn_2024.csv"),
}

MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models", "2024 models", "with socio")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "2024 data", "with socio")


def generate_season_hte(season_name: str, input_csv: str) -> pd.DataFrame:
    """Trains CausalForestDML models and computes row-level HTEs for a season."""
    print(f"\n=======================================================")
    print(f"Generating Row-Level HTE for {season_name.upper()} 2024")
    print(f"Source file: {input_csv}")
    print(f"=======================================================")

    df = pd.read_csv(input_csv)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} records (initial: {initial_len}).")

    Y = df["wdLoad"].values

    # ----------------------------------------------------
    # 1. Weather CausalForestDML (T = [temp, rh])
    # ----------------------------------------------------
    print("Fitting Weather CausalForestDML (T=[temp, rh], X=[hour, weekend, holiday, festival, office_hours])...")
    t0 = time.time()
    T_weather = df[["temp", "rh"]].values
    X_weather = df[["hour", "weekend", "holiday", "festival", "office_hours"]].values

    weather_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        model_t=MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        n_estimators=500,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    weather_model.fit(Y, T_weather, X=X_weather)
    weather_effects = weather_model.const_marginal_effect(X_weather)
    df["temp_hte"] = weather_effects[:, 0]
    df["rh_hte"] = weather_effects[:, 1]
    print(f"  -> Weather HTE completed in {time.time() - t0:.1f}s")
    print(f"  -> Mean ATE (Temp): {df['temp_hte'].mean():.4f} MW/°C")
    print(f"  -> Mean ATE (RH):   {df['rh_hte'].mean():.4f} MW/%")

    # ----------------------------------------------------
    # 2. Calendar CausalForestDML (T = [weekend, holiday, festival, office_hours])
    # ----------------------------------------------------
    print("Fitting Calendar CausalForestDML (T=[weekend, holiday, festival, office_hours], X=[temp, rh, hour])...")
    t1 = time.time()
    T_cal = df[["weekend", "holiday", "festival", "office_hours"]].values
    X_cal = df[["temp", "rh", "hour"]].values

    cal_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        model_t=MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        n_estimators=500,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    cal_model.fit(Y, T_cal, X=X_cal)
    cal_effects = cal_model.const_marginal_effect(X_cal)
    df["weekend_hte"] = cal_effects[:, 0]
    df["holiday_hte"] = cal_effects[:, 1]
    df["festival_hte"] = cal_effects[:, 2]
    df["office_hte"] = cal_effects[:, 3]
    print(f"  -> Calendar HTE completed in {time.time() - t1:.1f}s")
    print(f"  -> Mean ATE (Weekend):      {df['weekend_hte'].mean():.4f} MW")
    print(f"  -> Mean ATE (Holiday):      {df['holiday_hte'].mean():.4f} MW")
    print(f"  -> Mean ATE (Festival):     {df['festival_hte'].mean():.4f} MW")
    print(f"  -> Mean ATE (Office Hours): {df['office_hte'].mean():.4f} MW")

    # ----------------------------------------------------
    # 3. Save CSV Outputs
    # ----------------------------------------------------
    # Weather-only HTE output (exact replica of winter_2024_with_hte.csv)
    weather_cols = [c for c in df.columns if c not in ["weekend_hte", "holiday_hte", "festival_hte", "office_hte"]]
    out_weather_models = os.path.join(MODELS_DIR, f"{season_name}_2024_with_hte.csv")
    out_weather_data = os.path.join(DATA_DIR, f"{season_name}_2024_with_hte.csv")
    df[weather_cols].to_csv(out_weather_models, index=False)
    df[weather_cols].to_csv(out_weather_data, index=False)

    # Full weather + calendar HTE output (exact replica of winter_2024_with_all_hte.csv)
    out_all_models = os.path.join(MODELS_DIR, f"{season_name}_2024_with_all_hte.csv")
    out_all_data = os.path.join(DATA_DIR, f"{season_name}_2024_with_all_hte.csv")
    df.to_csv(out_all_models, index=False)
    df.to_csv(out_all_data, index=False)

    print(f"Saved: {out_weather_models}")
    print(f"Saved: {out_all_models}")
    return df


def run_all_seasons():
    """Generates row-level HTEs across Winter, Summer, Monsoon, and Autumn."""
    start_total = time.time()
    results = {}
    for season_name, csv_path in INPUT_SEASONS.items():
        results[season_name] = generate_season_hte(season_name, csv_path)

    print("\n=======================================================")
    print(f"All 4 Seasons Row-Level HTE Generation Complete! ({time.time() - start_total:.1f}s)")
    print("=======================================================")
    for s_name, res_df in results.items():
        print(f"  - {s_name.capitalize()}: {len(res_df)} tuples with temp_hte, rh_hte, weekend_hte, holiday_hte, festival_hte, office_hte")


if __name__ == "__main__":
    run_all_seasons()
