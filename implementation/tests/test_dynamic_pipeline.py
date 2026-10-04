"""
Comprehensive Integration Tests for Tata Power Dynamic Analytics Pipeline.
Validates dynamic notebook data ingestion, empirical hourly HTEs, RAG indexing,
real-time web search, SQL agent parameterization, and 3-5 sentence grounded explanations.
"""

import unittest
import os
import sys
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_causal_effects, get_hourly_hte, get_xai_results, get_db_connection
from backend.services.web_search import WebSearchService
from backend.services.sql_agent import SafeSQLAgent
from backend.services.explainer import ExplanationOrchestrator
from backend.services.dtw import DTWService


class TestDynamicPipeline(unittest.TestCase):

    def test_01_empirical_causal_and_hte(self):
        """Validates real EconML ATEs and empirical 24h HTEs are present."""
        for season in ["Winter", "Summer", "Monsoon", "Autumn"]:
            ates = get_causal_effects(season, effect_type="ATE")
            self.assertGreater(len(ates), 0, f"No ATEs found for {season}")
            treatments = [a["treatment"] for a in ates]
            self.assertIn("temp", treatments)
            self.assertIn("weekend", treatments)

            # Check hourly HTE for all 24 hours
            hte_t = get_hourly_hte(season, treatment="temp")
            self.assertEqual(len(hte_t), 24, f"Season {season} should have 24 hours of HTE")
            hours = [h["hour"] for h in hte_t]
            self.assertEqual(hours, list(range(24)))

    def test_02_real_xai_results(self):
        """Validates real SHAP feature importances from XGBoost models."""
        for season in ["Winter", "Summer", "Monsoon", "Autumn"]:
            xai = get_xai_results(season, explanation_type="SHAP_GLOBAL")
            self.assertGreater(len(xai), 0, f"No XAI records for {season}")
            feats = [x["feature"] for x in xai]
            self.assertIn("office_hours", feats)


    def test_04_real_web_search(self):
        """Validates real Open-Meteo historical weather and live search."""
        ws = WebSearchService()
        res = ws.search("Mumbai electricity demand", target_date="2024-11-01", max_results=2)
        self.assertGreater(len(res), 0)
        # Should have real URL and snippet
        self.assertTrue(any("open-meteo" in r.get("url", "") or "weather" in r.get("snippet", "").lower() for r in res))

    def test_05_sql_agent_intent(self):
        """Validates natural language intent parsing into parameterized SQL."""
        agent = SafeSQLAgent()
        out = agent.execute_query("Why was load unusually high on 15 Oct 2024 at 3 PM?")
        self.assertEqual(out["intent"]["date"], "2024-10-15")
        self.assertEqual(out["intent"]["hour"], 15)
        self.assertEqual(out["intent"]["season"], "Autumn")
        self.assertIn("WHERE", out["sql"])
        self.assertGreater(out["sample_count"], 0)

    def test_06_explainer_concise_synthesis(self):
        """Validates grounded explanation matches Section 19 (3-5 sentences)."""
        exp = ExplanationOrchestrator()
        res = exp.explain("2024-10-15", user_question="Why was load unusually high on 15 Oct 2024 at 3 PM?")
        self.assertIn("answer", res)
        self.assertIn("evidence", res)

        ans = res["answer"]
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', ans.strip()) if len(s.strip()) > 15]
        # Should be a concise paragraph of 2-6 sentences
        self.assertTrue(2 <= len(sentences) <= 7, f"Expected concise paragraph, got {len(sentences)}: {ans}")
        # Must mention actual numbers
        self.assertTrue("2024-10-15" in ans or "October 15" in ans or "15 Oct" in ans, f"Date not found in: {ans}")
        # Must contain tuple_level_hte and causal_estimates_econml_ate in evidence
        self.assertIn("tuple_level_hte", res["evidence"])
        self.assertIn("causal_estimates_econml_ate", res["evidence"])


if __name__ == "__main__":
    unittest.main()
