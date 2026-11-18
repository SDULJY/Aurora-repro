"""Phase 1 观测解析的轻量单元测试。

运行：

    python -m unittest tests.test_obs_sources -v
"""

import unittest

import numpy as np

from obs_sources import value_at


class TestValueAt(unittest.TestCase):
    def setUp(self):
        self.records = {
            "time": np.array([0, 6, 12], dtype=np.int64),
            "2t": np.array([270.0, 271.0, 272.0], dtype=np.float32),
            "msl": np.array([1000.0, 1001.0, 1002.0], dtype=np.float32),
        }

    def test_exact_hour(self):
        self.assertAlmostEqual(value_at(self.records, 6, "2t"), 271.0)

    def test_nearest_half_hour(self):
        # 5.5h 与 6h 相差 0.5h，仍在容差内
        self.assertAlmostEqual(value_at(self.records, 6, "2t", tolerance=0.5), 271.0)

    def test_outside_tolerance(self):
        self.assertTrue(np.isnan(value_at(self.records, 9, "2t", tolerance=0.5)))

    def test_missing_variable(self):
        self.assertTrue(np.isnan(value_at(self.records, 0, "10u")))


if __name__ == "__main__":
    unittest.main()
