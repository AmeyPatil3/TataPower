"""
Unit tests for Safe Parameter-to-SQL Agent.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.sql_agent import SafeSQLAgent, QueryIntent


class TestSafeSQLAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SafeSQLAgent()

    def test_average_load_summer_festivals(self):
        """Test question from specification: 'What was the average load during summer festival days in 2024?'"""
        q = "What was the average load during summer festival days in 2024?"
        res = self.agent.execute_query(q)

        self.assertEqual(res["intent"]["metric"], "AVG")
        self.assertEqual(res["intent"]["variable"], "wdLoad")
        self.assertEqual(res["intent"]["season"], "Summer")
        self.assertEqual(res["intent"]["festival"], 1)
        self.assertEqual(res["intent"]["year"], 2024)
        self.assertIn("SELECT ROUND(AVG(wdLoad), 2)", res["sql"])
        self.assertIn("WHERE season = ? AND year = ? AND festival = ?", res["sql"])
        self.assertIsNotNone(res["value"])
        self.assertGreater(res["sample_count"], 0)

    def test_peak_temperature_monsoon(self):
        q = "What was the peak temperature in monsoon?"
        res = self.agent.execute_query(q)

        self.assertEqual(res["intent"]["metric"], "MAX")
        self.assertEqual(res["intent"]["variable"], "temp")
        self.assertEqual(res["intent"]["season"], "Monsoon")
        self.assertIsNotNone(res["value"])

    def test_group_by_hour(self):
        q = "Average load by hour in winter"
        res = self.agent.execute_query(q)

        self.assertEqual(res["intent"]["group_by"], "hour")
        self.assertEqual(len(res["records"]), 24)

    def test_safety_disallows_injection(self):
        """Malicious string inputs cannot alter SQL query structure due to intent extraction."""
        malicious = "Show load; DROP TABLE load_data; --"
        res = self.agent.execute_query(malicious)

        self.assertNotIn("DROP", res["sql"])
        self.assertTrue(res["sql"].startswith("SELECT"))


if __name__ == "__main__":
    unittest.main()
