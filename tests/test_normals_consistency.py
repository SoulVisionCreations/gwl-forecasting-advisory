"""neighbours.restrict — limit the normals NeighbourSet to the forecast's wells so forecast and
normal are an apples-to-apples IDW blend (same stations + weights)."""
import unittest

import pandas as pd

from advisory.neighbours import NeighbourSet, restrict


def _nb():
    return NeighbourSet(
        codes=["A", "B", "C", "D"],
        weights={"A": 1.0, "B": 0.5, "C": 0.25, "D": 0.1},
        distances={"A": 1.0, "B": 2.0, "C": 4.0, "D": 10.0},
        series={"A": pd.Series([1.0]), "B": pd.Series([2.0]), "C": pd.Series([3.0])},  # D: no history
        nearest_km=1.0, n_with_data=3, fallback_codes={"B"},
    )


class TestRestrict(unittest.TestCase):

    def test_keeps_only_requested_preserving_order(self):
        r = restrict(_nb(), {"C", "A", "D"})     # note: unordered input set
        self.assertEqual(r.codes, ["A", "C", "D"], "order follows the original nb.codes")
        self.assertEqual(set(r.weights), {"A", "C", "D"})
        self.assertNotIn("B", r.series)

    def test_n_with_data_counts_only_kept_with_history(self):
        r = restrict(_nb(), {"A", "C", "D"})     # D has no series
        self.assertEqual(r.n_with_data, 2)       # A, C only

    def test_nearest_km_recomputed_from_kept(self):
        r = restrict(_nb(), {"C", "D"})          # nearest kept = C @ 4.0
        self.assertEqual(r.nearest_km, 4.0)

    def test_fallback_codes_intersected(self):
        r = restrict(_nb(), {"A", "C"})          # B (a fallback) is not kept
        self.assertEqual(r.fallback_codes, set())

    def test_empty_intersection(self):
        r = restrict(_nb(), {"Z"})
        self.assertEqual(r.codes, [])
        self.assertEqual(r.n_with_data, 0)
        self.assertIsNone(r.nearest_km)


if __name__ == "__main__":
    unittest.main(verbosity=2)
