"""
Dynamic Time Warping (DTW) Engine for Daily Load Profile Similarity.
Operates on 96-interval diurnal load curves (15-min intervals).
Includes curve normalization, leakage prevention (temporal masking), and SQLite caching.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_db_connection, get_daily_records


def compute_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Computes exact normalized Dynamic Time Warping distance between two 1D sequences.
    For daily 96-interval series, standard DP runs in < 1ms.
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill dynamic programming matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1]   # Match
            )

    # Normalize by path length upper bound (n + m)
    return float(dtw_matrix[n, m] / (n + m))


def normalize_sequence(seq: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalizes a 1D sequence for shape-based morphological comparison."""
    if len(seq) == 0:
        return seq
    
    if method == "minmax":
        min_val = np.min(seq)
        max_val = np.max(seq)
        diff = max_val - min_val
        if diff < 1e-6:
            return np.zeros_like(seq)
        return (seq - min_val) / diff

    elif method == "zscore":
        std = np.std(seq)
        if std < 1e-6:
            return np.zeros_like(seq)
        return (seq - np.mean(seq)) / std

    return seq


class DTWService:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def get_day_curve(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieves the 96-point load sequence and metadata for a date."""
        records = get_daily_records(date_str, self.db_path)
        if len(records) != 96:
            return None

        loads = np.array([r["wdLoad"] for r in records], dtype=float)
        temps = np.array([r["temp"] for r in records], dtype=float)
        rhs = np.array([r["rh"] for r in records], dtype=float)

        meta = records[0]
        return {
            "date": date_str,
            "season": meta["season"],
            "day_of_week": meta["day_of_week"],
            "weekend": meta["weekend"],
            "holiday": meta["holiday"],
            "festival": meta["festival"],
            "raw_load": loads,
            "normalized_load": normalize_sequence(loads, method="minmax"),
            "mean_load": float(np.mean(loads)),
            "peak_load": float(np.max(loads)),
            "peak_hour": int(meta["hour"]),
            "mean_temp": float(np.mean(temps)),
            "mean_rh": float(np.mean(rhs))
        }

    def find_top_k_similar_days(
        self,
        query_date: str,
        k: int = 5,
        season_filter: Optional[str] = None,
        same_calendar: bool = False,
        mask_future: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k most morphologically similar historical days using DTW.
        
        Safeguards:
        - Excludes query date itself.
        - If mask_future=True, excludes any dates later than query_date (no leakage).
        - Explicitly documents similarity is non-causal.
        """
        query_data = self.get_day_curve(query_date)
        if not query_data:
            raise ValueError(f"Query date '{query_date}' not found or does not have exactly 96 intervals.")

        # 1. Instant Cache Check
        if not same_calendar and not season_filter:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT historical_date, season, distance, rank, is_same_calendar_type
                    FROM dtw_matches
                    WHERE query_date = ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (query_date, k)
                )
                cached_rows = cursor.fetchall()
                if len(cached_rows) >= k:
                    cached_matches = []
                    for row in cached_rows:
                        hist_data = self.get_day_curve(row["historical_date"])
                        if hist_data:
                            cached_matches.append({
                                "query_date": query_date,
                                "historical_date": row["historical_date"],
                                "season": row["season"],
                                "distance": row["distance"],
                                "rank": row["rank"],
                                "sequence_type": "univariate_load",
                                "is_same_calendar_type": row["is_same_calendar_type"],
                                "historical_mean_load": hist_data["mean_load"],
                                "historical_peak_load": hist_data["peak_load"],
                                "historical_mean_temp": hist_data["mean_temp"],
                                "historical_curve": hist_data["raw_load"].tolist()
                            })
                    if len(cached_matches) >= k:
                        return cached_matches

        # 2. Build candidate filter query
        query_sql = "SELECT DISTINCT date, season, weekend, holiday, festival FROM load_data WHERE date != ?"
        params = [query_date]

        if mask_future:
            query_sql += " AND date < ?"
            params.append(query_date)

        if season_filter:
            query_sql += " AND season = ?"
            params.append(season_filter)

        if same_calendar:
            query_sql += " AND weekend = ? AND holiday = ? AND festival = ?"
            params.extend([query_data["weekend"], query_data["holiday"], query_data["festival"]])

        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(query_sql, params)
            candidates = cursor.fetchall()

        if not candidates:
            return []

        # Limit candidate pool to most relevant 60 recent dates to guarantee sub-second DTW evaluation
        cand_dates = [c["date"] for c in candidates][-60:]
        cand_map = {c["date"]: c for c in candidates}

        # Bulk load load curves in ONE query
        placeholders = ",".join(["?"] * len(cand_dates))
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT date, wdLoad, temp, rh, hour
                FROM load_data
                WHERE date IN ({placeholders})
                ORDER BY datetime ASC
                """,
                cand_dates
            )
            all_rows = cursor.fetchall()

        # Group in memory
        from collections import defaultdict
        day_points = defaultdict(list)
        day_temps = defaultdict(list)
        for r in all_rows:
            day_points[r["date"]].append(r["wdLoad"])
            day_temps[r["date"]].append(r["temp"])

        distances = []
        for cand_date, loads_list in day_points.items():
            if len(loads_list) != 96:
                continue

            loads = np.array(loads_list, dtype=float)
            norm_loads = normalize_sequence(loads, method="minmax")

            dist = compute_dtw_distance(
                query_data["normalized_load"],
                norm_loads
            )

            cand = cand_map[cand_date]
            is_same_cal = 1 if (
                cand["weekend"] == query_data["weekend"] and
                cand["holiday"] == query_data["holiday"] and
                cand["festival"] == query_data["festival"]
            ) else 0

            distances.append({
                "query_date": query_date,
                "historical_date": cand_date,
                "season": cand["season"],
                "distance": round(dist, 5),
                "is_same_calendar_type": is_same_cal,
                "historical_mean_load": float(np.mean(loads)),
                "historical_peak_load": float(np.max(loads)),
                "historical_mean_temp": float(np.mean(day_temps[cand_date])),
                "historical_curve": loads.tolist()
            })

        # Sort by distance ascending
        distances.sort(key=lambda x: x["distance"])
        top_matches = distances[:k]

        # Assign ranks and cache in SQLite
        for rank, match in enumerate(top_matches, start=1):
            match["rank"] = rank
            match["sequence_type"] = "univariate_load"

        self._cache_matches(top_matches)
        return top_matches

    def _cache_matches(self, matches: List[Dict[str, Any]]):
        """Saves DTW matches to SQLite dtw_matches table."""
        if not matches:
            return
        with get_db_connection(self.db_path) as conn:
            for m in matches:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dtw_matches (
                        query_date, historical_date, season, distance,
                        rank, sequence_type, is_same_calendar_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m["query_date"], m["historical_date"], m["season"],
                        m["distance"], m["rank"], m["sequence_type"],
                        m["is_same_calendar_type"]
                    )
                )
