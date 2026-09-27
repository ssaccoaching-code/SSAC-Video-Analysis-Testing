import json
import unittest

from core.coach import coach
from core.report import to_json, to_text
from core.types import Result


class CoachTests(unittest.TestCase):
    def test_nothing_measured(self) -> None:
        advice = coach([Result(module="dive", metrics={"reaction_time_s": None})])
        self.assertEqual(len(advice.summary), 1)
        self.assertEqual(advice.plan, [])

    def test_slow_reaction_makes_a_plan_item(self) -> None:
        advice = coach([Result(module="dive", metrics={"reaction_time_s": 0.92, "entry_angle_deg": 38.0})])
        self.assertTrue(any("reaction" in p["focus"].lower() for p in advice.plan))
        self.assertTrue(any("38" in s for s in advice.summary))
        self.assertIn("Tentative", advice.basis)

    def test_event_fade_is_the_first_item(self) -> None:
        event = Result(module="event_above", metrics={
            "lengths": 4, "distance_m": 91.4, "event_time_s": 60.0, "avg_speed_mps": 1.5,
            "split_by_length_s": [13.0, 15.0, 16.0, 16.5], "speed_change_pct": -10.0,
            "tempo_change_pct": -2.0, "turn_time_by_turn_s": [6.0, 6.2, 6.1], "turn_time_avg_s": 6.1,
        })
        advice = coach([event])
        self.assertEqual(advice.plan[0]["focus"], "Pacing and finish")
        self.assertTrue(advice.summary[0].startswith("Whole event: 4 lengths"))

    def test_details_feed_the_summary_and_the_json(self) -> None:
        event = Result(module="event_underwater", metrics={"lengths": 2, "underwater_distance_by_length_m": [3.0, 3.5]},
                       details=[Result(module="turn_L1", metrics={"turn_time_s": 8.0, "turn_zone_m": 5.0})])
        advice = coach([event])
        self.assertTrue(any(p["focus"] == "Faster turns" for p in advice.plan))
        self.assertTrue(any(p["focus"] == "Longer underwaters" for p in advice.plan))
        payload = json.loads(to_json([event]))
        self.assertIn("advice", payload)
        self.assertEqual(payload["results"][0]["details"][0]["module"], "turn_L1")
        text = to_text([event])
        self.assertIn("[turn_L1]", text)
        self.assertIn("[improvement plan, tentative]", text)


if __name__ == "__main__":
    unittest.main()
