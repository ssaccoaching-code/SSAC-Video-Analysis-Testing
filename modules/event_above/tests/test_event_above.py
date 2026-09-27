"""Tests for the event_above module.

Run with:  python -m pytest modules/event_above

The contract tests come from tests/helpers.py and must keep passing. The
event test builds a 4-length cartoon clip (tools/make_sample_clip.py) and
checks that the lengths, turns and splits come out.
"""

import unittest

from tests.helpers import ModuleContractTest, moving_dot_clip, synthetic_event_clip
from modules.event_above.module import EventAboveModule


class EventAboveModuleContractTest(ModuleContractTest, unittest.TestCase):
    module_class = EventAboveModule


class EventAboveModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = EventAboveModule()

    def test_short_clip_reports_nothing_but_does_not_crash(self) -> None:
        result = self.module.run(moving_dot_clip(), {})
        self.assertEqual(result.module, "event_above")

    def test_four_length_event(self) -> None:
        clip = synthetic_event_clip(lengths=4, seconds=40.0)
        if clip is None:
            self.skipTest("no video encoder available")
        with clip:
            result = self.module.run(clip, {"start_s": 1.0, "course": "SCY"})
        self.assertEqual(result.metrics["lengths"], 4)
        self.assertEqual(result.metrics["turn_count"], 3)
        self.assertEqual(len(result.metrics["split_by_length_s"]), 4)
        for split in result.metrics["split_by_length_s"]:
            self.assertTrue(7.0 < split < 12.0, split)
        self.assertAlmostEqual(result.metrics["event_time_s"], sum(result.metrics["split_by_length_s"]), delta=0.2)
        self.assertEqual(len(result.metrics["turn_time_by_turn_s"]), 3)
        self.assertIn("wall_1", result.key_frames)
        self.assertIn("finish", result.key_frames)
        names = [d.module for d in result.details]
        self.assertEqual(names[:3], ["underwater_L1", "swim_L1", "turn_L1"])
        self.assertEqual(len(names), 11)


if __name__ == "__main__":
    unittest.main()
