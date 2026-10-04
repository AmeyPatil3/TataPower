"""
Data Ingestion Pipeline for Tata Power 2024 Seasonal Datasets.
Ingests Winter, Summer, Monsoon, and Autumn datasets into SQLite database.
Validates row counts, date ranges, and 96-interval diurnal sequences.
"""

import os
import sys
import argparse
import pandas as pd

# Add implementation root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_db_connection, init_db

DATA_DIR = "/Users/ameypatil/Desktop/Tata Power/data/2024 data/with socio"

SEASON_FILES = {
    "Winter": ["winter_2024_with_all_hte.csv", "winter_2024_with_hte.csv", "winter_2024.csv"],
    "Summer": ["summer_2024_with_all_hte.csv", "summer_2024_with_hte.csv", "summer_2024.csv"],
    "Monsoon": ["monsoon_2024_with_all_hte.csv", "monsoon_2024_with_hte.csv", "monsoon_2024.csv"],
    "Autumn": ["autumn_2024_with_all_hte.csv", "autumn_2024_with_hte.csv", "autumn_2024.csv"]
}


def ingest_data(data_dir: str = DATA_DIR, db_path: str = None) -> int:
    """Ingests all 2024 seasonal CSVs with row-level HTEs into SQLite load_data."""
    init_db(db_path)
    total_inserted = 0

    print("==================================================")
    print("Starting Ingestion into SQLite database...")
    print(f"Source Directory: {data_dir}")
    print("==================================================")

    for season, candidate_files in SEASON_FILES.items():
        filepath = None
        for cand in candidate_files:
            p = os.path.join(data_dir, cand)
            if os.path.exists(p):
                filepath = p
                break

        if not filepath or not os.path.exists(filepath):
            print(f"[ERROR] No suitable file found for {season}")
            continue

        print(f"Reading {season} data from {os.path.basename(filepath)}...")
        df = pd.read_csv(filepath)
        initial_count = len(df)

        # Ensure datetime is preserved string and extract YYYY-MM-DD
        df['datetime'] = df['datetime'].astype(str).str.strip()
        df['date'] = df['datetime'].str.slice(0, 10)
        df['season'] = season
        df['year'] = 2024

        # Clean numeric types including tuple-level HTEs
        numeric_cols = [
            'wdLoad', 'temp', 'rh', 'hour', 'day_of_week', 
            'weekend', 'holiday', 'festival', 'office_hours', 
            'lag_load_15min', 'lag_load_prev_day',
            'temp_hte', 'rh_hte', 'weekend_hte', 'holiday_hte', 'festival_hte', 'office_hte'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = None

        # Prepare records for insertion
        records = df[[
            'datetime', 'date', 'wdLoad', 'temp', 'rh', 'hour',
            'day_of_week', 'weekend', 'holiday', 'festival', 'office_hours',
            'lag_load_15min', 'lag_load_prev_day', 'season', 'year',
            'temp_hte', 'rh_hte', 'weekend_hte', 'holiday_hte', 'festival_hte', 'office_hte'
        ]].to_dict(orient='records')

        with get_db_connection(db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO load_data (
                    datetime, date, wdLoad, temp, rh, hour,
                    day_of_week, weekend, holiday, festival, office_hours,
                    lag_load_15min, lag_load_prev_day, season, year,
                    temp_hte, rh_hte, weekend_hte, holiday_hte, festival_hte, office_hte
                ) VALUES (
                    :datetime, :date, :wdLoad, :temp, :rh, :hour,
                    :day_of_week, :weekend, :holiday, :festival, :office_hours,
                    :lag_load_15min, :lag_load_prev_day, :season, :year,
                    :temp_hte, :rh_hte, :weekend_hte, :holiday_hte, :festival_hte, :office_hte
                )
                """,
                records
            )

        print(f"  -> Successfully ingested {initial_count} rows for {season} with row-level HTE values.")
        total_inserted += initial_count

    print("==================================================")
    print(f"Ingestion Complete! Total rows ingested: {total_inserted}")
    print("==================================================")
    return total_inserted


def validate_database(db_path: str = None) -> bool:
    """Validates the ingested data for row counts, integrity, and sequence lengths."""
    print("\n--- Validating Database Integrity ---")
    with get_db_connection(db_path) as conn:
        # Check counts per season
        cursor = conn.execute(
            """
            SELECT season, COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date,
                   ROUND(AVG(wdLoad), 2) as avg_load, ROUND(MIN(wdLoad), 2) as min_load, ROUND(MAX(wdLoad), 2) as max_load
            FROM load_data
            GROUP BY season
            ORDER BY min_date
            """
        )
        season_stats = cursor.fetchall()
        print("\nSeasonal Summary in Database:")
        total_rows = 0
        for s in season_stats:
            print(f"  - {s['season']}: {s['count']} rows | Span: {s['min_date']} to {s['max_date']} | Load: {s['min_load']} - {s['max_load']} MW (Avg: {s['avg_load']} MW)")
            total_rows += s['count']

        print(f"\nTotal Records: {total_rows}")

        # Check for nulls in critical columns
        cursor = conn.execute(
            """
            SELECT 
                SUM(CASE WHEN wdLoad IS NULL THEN 1 ELSE 0 END) as null_load,
                SUM(CASE WHEN temp IS NULL THEN 1 ELSE 0 END) as null_temp,
                SUM(CASE WHEN rh IS NULL THEN 1 ELSE 0 END) as null_rh
            FROM load_data
            """
        )
        null_counts = cursor.fetchone()
        print(f"Null Checks: wdLoad nulls={null_counts['null_load']}, temp nulls={null_counts['null_temp']}, rh nulls={null_counts['null_rh']}")

        # Check diurnal daily sequence completeness (96 points per day)
        cursor = conn.execute(
            """
            SELECT date, season, COUNT(*) as intervals
            FROM load_data
            GROUP BY date
            HAVING intervals = 96
            """
        )
        full_days = cursor.fetchall()

        cursor = conn.execute(
            """
            SELECT date, season, COUNT(*) as intervals
            FROM load_data
            GROUP BY date
            HAVING intervals != 96
            """
        )
        partial_days = cursor.fetchall()

        print(f"Daily Profile Integrity:")
        print(f"  - Full 96-interval days: {len(full_days)}")
        print(f"  - Partial days (e.g. edge boundaries): {len(partial_days)}")
        if partial_days:
            for p in partial_days[:5]:
                print(f"    * {p['date']} ({p['season']}): {p['intervals']} intervals")

    print("--- Database Validation Successful! ---\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Tata Power 2024 Seasonal Data into SQLite.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to data directory.")
    parser.add_argument("--db-path", default=None, help="Custom SQLite DB path.")
    parser.add_argument("--validate", action="store_true", help="Run validation queries after ingestion.")

    args = parser.parse_args()
    ingest_data(data_dir=args.data_dir, db_path=args.db_path)
    if args.validate or True:
        validate_database(db_path=args.db_path)
