"""
XAI Service Wrapper for SHAP feature importances and LIME local explanations.
"""

import os
import sys
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_xai_results


class XAIService:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def get_global_importance(self, season: str) -> List[Dict[str, Any]]:
        """Returns sorted global SHAP feature importances."""
        return get_xai_results(season, explanation_type="SHAP_GLOBAL", db_path=self.db_path)
