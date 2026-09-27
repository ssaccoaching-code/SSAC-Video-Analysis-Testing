import unittest

import numpy as np

from core.calibrate import Calibration, parse_points
from core.signal import count_rhythm, crossing_time, fill_short_gaps, moving_average, per_second


class SignalTests(unittest.TestCase):
    def test_moving_average_ignores_nan_and_short_arrays(self) -> None:
        values = np.array([1.0, np.nan, 3.0])
        out = moving_average(values, 3)
        self.assertAlmostEqual(out[1], 2.0)
        self.assertEqual(moving_average(np.array([1.0, 2.0]), 9).size, 2)

    def test_count_rhythm_finds_a_sine(self) -> None:
        fps = 30.0
        t = np.arange(0, 8, 1 / fps)
        signal = np.sin(2 * np.pi * t / 0.9) + 0.1 * np.random.default_rng(1).normal(size=t.size)
        count, peaks, period = count_rhythm(signal, fps, 0.35, 1.6)
        self.assertIsNotNone(period)
        self.assertAlmostEqual(period, 0.9, delta=0.1)
        self.assertAlmostEqual(count, 8 / 0.9, delta=1.5)

    def test_crossing_time_interpolates(self) -> None:
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 2.0, 4.0])
        self.assertAlmostEqual(crossing_time(t, v, 1.0), 0.5)
        self.assertIsNone(crossing_time(t, v, 9.0))

    def test_fill_short_gaps_keeps_long_gaps(self) -> None:
        v = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
        out = fill_short_gaps(v, 1)
        self.assertAlmostEqual(out[1], 2.0)
        self.assertTrue(np.isnan(out[3]) and np.isnan(out[5]))

    def test_per_second(self) -> None:
        t = np.arange(0, 3, 0.5)
        v = np.array([1, 1, 2, 2, 3, 3], dtype=float)
        self.assertEqual(per_second(t, v), [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])


class CalibrationTests(unittest.TestCase):
    def test_two_points_map_linearly_and_extrapolate(self) -> None:
        cal = Calibration([(100.0, 0.0), (1100.0, 10.0)])
        self.assertAlmostEqual(cal.x_to_m(600.0), 5.0)
        self.assertAlmostEqual(cal.x_to_m(1600.0), 15.0)
        self.assertAlmostEqual(cal.m_per_px(), 0.01)
        self.assertEqual(cal.direction, 1)
        self.assertEqual(cal.wall_x, 100.0)

    def test_reverse_direction(self) -> None:
        cal = Calibration([(1100.0, 0.0), (100.0, 10.0)])
        self.assertEqual(cal.direction, -1)
        self.assertAlmostEqual(cal.x_to_m(600.0), 5.0)

    def test_parse_points(self) -> None:
        self.assertEqual(parse_points("10:0, 20:5"), [(10.0, 0.0), (20.0, 5.0)])
        self.assertEqual(parse_points(None), [])

    def test_round_trip_dict(self) -> None:
        cal = Calibration([(0.0, 0.0), (10.0, 1.0)], method="points", confidence="high")
        again = Calibration.from_dict(cal.to_dict())
        self.assertEqual(again.points, cal.points)
        self.assertEqual(again.confidence, "high")


if __name__ == "__main__":
    unittest.main()
