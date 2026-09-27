"""Whole event, underwater camera: every length of a race, however many.

Give it a clip of a whole swim from an underwater camera (any number of
lengths) and it finds the lengths itself, runs the phase modules on each one
and lines the numbers up by length. Underwater footage is best for what
happens off the walls: the entry, the kicks, the breakout and the push-off.

Metrics:
  lengths                      how many lengths were found
  distance_m                   lengths x pool length
  event_time_s                 start signal (or first movement) to the final wall touch
  avg_speed_mps                distance / event time
  split_by_length_s            time for each length, wall to wall
  speed_by_length_mps          pool length / split, per length
  speed_change_pct             last length speed vs the second length (or the first, in a 2-length event)
  underwater_distance_by_length_m   distance to the breakout after each start / push-off
  underwater_time_by_length_s
  underwater_change_pct        last length underwater distance vs the second (or first) length
  kicks_by_length              dolphin kicks per length
  breakout_speed_by_length_mps, breakout_slowdown_by_length_pct, peak_underwater_speed_by_length_mps
  turn_count, turn_time_by_turn_s, wall_contact_by_turn_s, push_off_speed_by_turn_mps
  strokes_by_length, tempo_by_length_spm   (rough from under the water)
  entry_depth_m, entry_angle_deg           from the dive, when the clip shows it

Key frames: "start", "wall_1", "wall_2", ..., "finish". The per-length
results (dive, underwater_L1, swim_L1, turn_L1, underwater_L2, ...) are kept
in result.details with their own key frames.

Extra options this module reads: the same as the phase modules
  (view, calib_points / calib_profile, turn_type, turn_zone_m).
"""

from __future__ import annotations

from typing import Any

from core.base import AnalysisModule
from core.event import analyze_event
from core.types import Result
from core.video import Clip


class EventUnderwaterModule(AnalysisModule):
    name = "event_underwater"
    description = "Whole event from an underwater camera, any number of lengths"
    owner = ""
    metric_keys = (
        "lengths",
        "distance_m",
        "event_time_s",
        "avg_speed_mps",
        "split_by_length_s",
        "speed_by_length_mps",
        "speed_change_pct",
        "underwater_distance_by_length_m",
        "underwater_time_by_length_s",
        "underwater_change_pct",
        "kicks_by_length",
        "breakout_speed_by_length_mps",
        "breakout_slowdown_by_length_pct",
        "peak_underwater_speed_by_length_mps",
        "turn_count",
        "turn_time_by_turn_s",
        "wall_contact_by_turn_s",
        "push_off_speed_by_turn_mps",
        "strokes_by_length",
        "tempo_by_length_spm",
        "entry_depth_m",
        "entry_angle_deg",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        event = analyze_event(clip, options, focus="underwater")
        for key in self.metric_keys:
            result.metrics[key] = event.metrics.get(key)
        result.key_frames.update(event.key_frames)
        result.notes.extend(event.notes)
        result.details = event.details
        return result
