"""
Analytical Pipeline Execution & Persistence.
Delegates to extract_and_populate to extract validated EconML ATEs,
empirical hourly HTEs, and XGBoost/SHAP feature attributions directly
from research notebooks and observation datasets.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.extract_and_populate import run_pipeline as extract_and_run
except ImportError:
    from extract_and_populate import run_pipeline as extract_and_run

def populate_baseline_causal_and_xai(db_path: str = None):
    return extract_and_run(db_path)

if __name__ == "__main__":
    extract_and_run()
