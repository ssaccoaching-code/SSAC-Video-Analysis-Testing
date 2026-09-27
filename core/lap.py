"""Whole-lap analysis: run every phase on one clip, in order, sharing one track.

    from core.lap import analyze_lap
    results = analyze_lap(clip, options)

The dive module finds the entry, the underwater module runs from there to the
breakout, the swim module from the breakout on, the turn module looks for a
wall touch-and-return, and if there is one the underwater module runs again
from the push-off. Modules that see nothing (a clip without a dive, say) just
say so in their notes; nothing crashes.
"""

from __future__ import annotations

from typing import Any

from core.base import CACHE_KEYS
from core.types import Result
from core.video import Clip
from modules.dive.module import DiveModule
from modules.swim.module import SwimModule
from modules.turn.module import TurnModule
from modules.underwater.module import UnderwaterModule


def analyze_lap(clip: Clip, options: dict[str, Any]) -> list[Result]:
    for key in CACHE_KEYS:
        options.setdefault(key, {})    # shared by every module below, and by the report
    results: list[Result] = []

    dive = DiveModule().run(clip, options)
    results.append(dive)
    entry_t = dive.metrics.get("entry_time_s")

    uw_options = dict(options)
    if entry_t is not None:
        uw_options["window_start_s"] = float(entry_t)
    underwater = UnderwaterModule().run(clip, uw_options)
    results.append(underwater)
    breakout_index = underwater.key_frames.get("breakout")

    turn = TurnModule().run(clip, options)
    wall_leave = turn.key_frames.get("wall_leave")

    swim_options = dict(options)
    if breakout_index is not None:
        swim_options["window_start_s"] = clip.frame_to_time(int(breakout_index))
    swim = SwimModule().run(clip, swim_options)
    results.append(swim)
    results.append(turn)

    if wall_leave is not None and turn.metrics.get("turn_time_s") is not None:
        second = dict(options)
        second["window_start_s"] = clip.frame_to_time(int(wall_leave))
        second["window_end_s"] = min(clip.duration_s, second["window_start_s"] + 8.0)
        after = UnderwaterModule().run(clip, second)
        after.module = "underwater_after_turn"
        results.append(after)
    return results
