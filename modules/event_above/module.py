"""Whole event, above-water camera: every length of a race, however many.

Give it a clip of a whole swim from an above-water camera (25, 50, 100, 200
yards or metres, any number of lengths) and it finds the lengths itself, runs
the phase modules on each one and lines the numbers up by length.

Metrics:
  lengths                      how many lengths were found
  distance_m                   lengths x pool length
  event_time_s                 start signal (or first movement) to the final wall touch
  avg_speed_mps                distance / event time
  split_by_length_s            time for each length, wall to wall (length 1 from the start)
  cumulative_split_s           running time at each wall
  speed_by_length_mps          pool length / split, per length
  speed_change_pct             last length speed vs the second length (or the first, in a 2-length event)
  strokes_by_length            strokes per length
  tempo_by_length_spm          strokes per minute, per length
  distance_per_stroke_by_length_m
  tempo_change_pct             last length tempo vs the second (or first) length
  strokes_change_pct           last length stroke count vs the second (or first) length
  underwater_distance_by_length_m   breakout distance after each start / push-off (rough from above)
  underwater_time_by_length_s
  turn_count, turn_time_by_turn_s, turn_time_avg_s   5 m in / 5 m out turn times
  wall_contact_by_turn_s, push_off_speed_by_turn_mps
  turn_type
  reaction_time_s, flight_time_s, entry_distance_m   from the dive, when the clip shows it

Key frames: "start", "wall_1", "wall_2", ..., "finish". The per-length
results (dive, underwater_L1, swim_L1, turn_L1, underwater_L2, ...) are kept
in result.details with their own key frames.

Extra options this module reads: the same as the phase modules
  (view, calib_points / calib_profile, markers, turn_type, turn_zone_m).
"""

from __future__ import annotations

from typing import Any

from core.base import AnalysisModule
from core.event import analyze_event
from core.types import Result
from core.video import Clip


class EventAboveModule(AnalysisModule):
    name = "event_above"
    description = "Whole event from an above-water camera, any number of lengths"
    owner = ""
    metric_keys = (
        "lengths",
        "distance_m",
        "event_time_s",
        "avg_speed_mps",
        "split_by_length_s",
        "cumulative_split_s",
        "speed_by_length_mps",
        "speed_change_pct",
        "strokes_by_length",
        "tempo_by_length_spm",
        "distance_per_stroke_by_length_m",
        "tempo_change_pct",
        "strokes_change_pct",
        "underwater_distance_by_length_m",
        "underwater_time_by_length_s",
        "turn_count",
        "turn_time_by_turn_s",
        "turn_time_avg_s",
        "wall_contact_by_turn_s",
        "push_off_speed_by_turn_mps",
        "turn_type",
        "reaction_time_s",
        "flight_time_s",
        "entry_distance_m",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        event = analyze_event(clip, options, focus="above")
        for key in self.metric_keys:
            result.metrics[key] = event.metrics.get(key)
        result.key_frames.update(event.key_frames)
        result.notes.extend(event.notes)
        result.details = event.details
        return result
