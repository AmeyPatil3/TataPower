"""
Causal Service Wrapper for EconML ATE and Diurnal Hourly HTE.
"""

import os
import sys
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_causal_effects, get_hourly_hte, get_day_hourly_hte


class CausalService:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def get_season_effects(self, season: str) -> Dict[str, Any]:
        """Retrieves all ATE and HTE effects for a season."""
        ate_records = get_causal_effects(season, effect_type="ATE", db_path=self.db_path)
        temp_hte = get_hourly_hte(season, treatment="temp", db_path=self.db_path)
        rh_hte = get_hourly_hte(season, treatment="rh", db_path=self.db_path)

        ates = {r["treatment"]: {
            "effect_value": r["effect_value"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"]
        } for r in ate_records}

        return {
            "season": season,
            "ate": ates,
            "hte_temp_hourly": temp_hte,
            "hte_rh_hourly": rh_hte
        }

    def get_hour_treatment_effect(self, season: str, treatment: str, hour: int) -> Optional[Dict[str, Any]]:
        """Retrieves heterogeneous treatment effect for a specific hour."""
        hte_list = get_hourly_hte(season, treatment=treatment, db_path=self.db_path)
        for h in hte_list:
            if h["hour"] == hour:
                return h
        return None

    def get_day_hourly_hte(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Retrieves observation-level hourly treatment effects for a single day,
        matching sociowinter_v2.ipynb Cell 13:
        Returns 24 hours of averages for Temperature, Relative Humidity,
        Weekend, Holiday, Festival, and Office Hours for the selected day.
        """
        return get_day_hourly_hte(date_str, self.db_path)

