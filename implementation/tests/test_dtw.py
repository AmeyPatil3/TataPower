"""
Unit tests for DTW Service and leakage safeguards.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.dtw import compute_dtw_distance, normalize_sequence, DTWService


class TestDTWService(unittest.TestCase):
    def test_identical_sequences_zero_distance(self):
        """Distance between identical curves must be 0.0."""
        s = np.sin(np.linspace(0, 2 * np.pi, 96))
        dist = compute_dtw_distance(s, s)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_shifted_sequence_distance(self):
        """Small phase shift should have low DTW distance."""
        s1 = np.sin(np.linspace(0, 2 * np.pi, 96))
        s2 = np.roll(s1, 2)  # slight phase shift
        dist = compute_dtw_distance(s1, s2)
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 0.1)

    def test_normalization_bounds(self):
        """Min-max normalized sequence must lie in [0, 1]."""
        raw = np.array([250.0, 300.0, 450.0, 520.0, 280.0])
        norm = normalize_sequence(raw, method="minmax")
        self.assertAlmostEqual(np.min(norm), 0.0)
        self.assertAlmostEqual(np.max(norm), 1.0)

    def test_find_similar_days_leakage_safeguard(self):
        """DTW matching with mask_future=True must never return candidate dates >= query_date."""
        svc = DTWService()
        query_date = "2024-05-15"
        matches = svc.find_top_k_similar_days(query_date=query_date, k=3, mask_future=True)

        self.assertTrue(len(matches) > 0)
        for m in matches:
            self.assertLess(m["historical_date"], query_date, "Leakage detected: candidate date is >= query date!")
            self.assertNotEqual(m["historical_date"], query_date)


if __name__ == "__main__":
    unittest.main()
