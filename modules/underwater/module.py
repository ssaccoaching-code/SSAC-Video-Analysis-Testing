"""Underwater module: from water entry (or push-off) to the breakout.

Metrics (from the team meeting):
  underwater_distance_m   distance covered before the head breaks the surface
  underwater_time_s       time spent underwater
  kick_count              number of dolphin kicks
  breakout_time_s         time of the breakout (from the start signal if --start is given, else in the clip)
  breakout_speed_mps      speed at the moment of breakout
  breakout_slowdown_pct   how much the swimmer slowed from the fastest underwater speed to the breakout (0 if none)
  peak_underwater_speed_mps  fastest underwater speed (after the entry glide settles)

Key frames: "entry", "first_kick", "breakout".

How it works: the underwater phase starts at the dive entry when the clip shows
one (the dive module's detector is reused), at the push-off after a turn, or at
the first frame the swimmer is seen. On an underwater camera the breakout is
the first frame the top of the body reaches the surface line. From above the
water it is the first burst of white foam that stays (arms breaking out).
Kicks are counted from the up-and-down oscillation of the body; from above the
water only the faint underwater shape is visible, so that count is marked as
rough.

Extra options this module reads:
  view=underwater|above     (auto-detected if not set)
  calib_points=x1:m1,x2:m2  (or calib_profile=name) for real distances
  from_wall_m=0             metres position of the wall this phase starts from (the event runner sets it)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.analysis import Prepared, prepare
from core.base import AnalysisModule
from core.signal import count_rhythm, moving_average
from core.types import Result
from core.video import Clip

BODY_AREA_FRACTION = 0.0004   # underwater the swimmer is a faint, smaller blob


def find_breakout(prep: Prepared, from_k: int, frame_area: float) -> int | None:
    """Track position of the breakout after from_k, or None."""
    tr = prep.track
    fps = tr.fps
    n = tr.times.size
    k_all = np.arange(n)
    later = k_all >= from_k + int(round(0.3 * fps))
    if not later.any():
        return None
    if prep.view == "underwater" and prep.surface_y is not None:
        at_surface = later & tr.found & (tr.top <= prep.surface_y + 0.02 * prep.background.shape[0])
        run = _first_run(at_surface, max(2, int(round(0.1 * fps))))
        if run is not None:
            return run
    white = moving_average(np.where(tr.found, tr.white, 0.0), max(3, int(round(0.15 * fps))))
    white = np.nan_to_num(white)
    # skip the entry splash: wait until the foam has died down once
    quiet = later & (white <= 0.15 * max(1.0, float(white[from_k : from_k + int(round(1.0 * fps))].max())))
    quiet_k = np.where(quiet)[0]
    if quiet_k.size == 0:
        return None
    later = k_all >= quiet_k[0]
    peak = float(white[later].max())
    if peak <= 0:
        return None
    strong = later & (white >= 0.4 * peak) & tr.found & (tr.area >= BODY_AREA_FRACTION * frame_area)
    return _first_run(strong, max(2, int(round(0.15 * fps))))


class UnderwaterModule(AnalysisModule):
    name = "underwater"
    description = "Underwater phase and breakout"
    owner = ""
    metric_keys = (
        "underwater_distance_m",
        "underwater_time_s",
        "kick_count",
        "breakout_time_s",
        "breakout_speed_mps",
        "breakout_slowdown_pct",
        "peak_underwater_speed_mps",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        window_start = float(options.get("window_start_s", 0.0))
        window_end = options.get("window_end_s")
        in_lap = "window_start_s" in options or window_end is not None
        prep = prepare(clip, options, phase="underwater", start_s=window_start, end_s=window_end)
        result.notes.extend(prep.notes)
        tr = prep.track
        fps = tr.fps
        n = tr.times.size
        k_all = np.arange(n)
        frame_area = clip.width * clip.height

        # where does the underwater phase start?
        start_k = None
        origin = "window start" if in_lap else "first sight of the swimmer"
        if not in_lap:
            from modules.dive.module import find_dive_events

            full = prepare(clip, options, phase="dive", full_frame=True)
            events = find_dive_events(full, frame_area)
            if events is not None and 0.05 < full.track.times[events["entry"]] - full.track.times[events["leave_block"]] < 2.0:
                entry_t = float(full.track.times[events["entry"]])
                start_k = int(np.argmin(np.abs(tr.times - entry_t)))
                origin = "dive entry"
                result.key_frames["entry"] = int(tr.indexes[start_k])
        if start_k is None:
            seen = np.where(tr.found & (tr.area >= BODY_AREA_FRACTION * frame_area))[0]
            if seen.size == 0:
                result.notes.append("underwater: swimmer not found in the clip")
                return result
            start_k = int(seen[0]) if not in_lap else 0
        start_t = float(tr.times[start_k])
        if origin != "dive entry":
            result.notes.append(f"underwater: phase measured from the {origin} ({start_t:.2f} s in the clip)")

        breakout_k = find_breakout(prep, start_k, frame_area)
        if breakout_k is None:
            result.notes.append("underwater: no breakout seen in the clip")
            return result
        breakout_t = float(tr.times[breakout_k])
        result.key_frames["breakout"] = int(tr.indexes[breakout_k])

        # times and distance
        signal_start = float(options.get("start_s", 0.0) or 0.0)
        if signal_start > 0 and breakout_t > signal_start:
            result.metrics["breakout_time_s"] = round(breakout_t - signal_start, 2)
        else:
            result.metrics["breakout_time_s"] = round(breakout_t, 2)
            result.notes.append("underwater: breakout_time_s is time in the clip (no --start given)")
        result.metrics["underwater_time_s"] = round(breakout_t - start_t, 2)
        from_wall = options.get("from_wall_m")
        if from_wall is not None:
            # the whole-event runner says which wall this length starts from
            distance = float(prep.x_m[breakout_k]) - float(from_wall)
        elif prep.cal.wall_x is not None or prep.cal.method.startswith("points") or prep.cal.method.startswith("profile"):
            distance = float(prep.x_m[breakout_k])
        else:
            distance = float(prep.x_m[breakout_k] - prep.x_m[start_k])
        if np.isfinite(distance):
            result.metrics["underwater_distance_m"] = round(abs(distance), 2)
            if prep.cal.confidence == "low":
                result.notes.append("underwater: distance uses a guessed scale; calibrate the camera for a real number")

        # speeds
        settle = start_k + int(round(0.4 * fps))
        uw = (k_all >= settle) & (k_all < breakout_k) & np.isfinite(prep.speed)
        around = (k_all >= breakout_k - int(round(0.15 * fps))) & (k_all <= breakout_k + int(round(0.15 * fps))) & np.isfinite(prep.speed)
        if around.any():
            b_speed = float(np.abs(prep.speed[around]).mean())
            result.metrics["breakout_speed_mps"] = round(b_speed, 2)
            if uw.sum() >= 3:
                peak_speed = float(np.percentile(np.abs(prep.speed[uw]), 90))
                result.metrics["peak_underwater_speed_mps"] = round(peak_speed, 2)
                if peak_speed > 0:
                    result.metrics["breakout_slowdown_pct"] = round(max(0.0, (peak_speed - b_speed) / peak_speed * 100.0), 1)

        # kicks: vertical oscillation of the body between entry and breakout
        kick_window = (k_all >= start_k + int(round(0.2 * fps))) & (k_all < breakout_k) & tr.found
        if kick_window.sum() >= int(0.5 * fps):
            if prep.view == "underwater":
                signal = np.where(kick_window, tr.bottom - tr.top, np.nan)
            else:
                signal = np.where(kick_window, tr.area, np.nan)
            sub = signal[kick_window]
            count, peaks, period = count_rhythm(sub, fps, min_period_s=0.28, max_period_s=1.0)
            result.metrics["kick_count"] = int(count)
            if peaks:
                first = int(np.where(kick_window)[0][peaks[0]])
                result.key_frames["first_kick"] = int(tr.indexes[first])
            if prep.view != "underwater":
                result.notes.append("underwater: kick count from above the water is rough; use the underwater camera for an exact count")
        else:
            result.notes.append("underwater: phase too short or swimmer not visible enough to count kicks")
        return result


def _first_run(flags: np.ndarray, length: int) -> int | None:
    run = 0
    for k, on in enumerate(flags):
        run = run + 1 if on else 0
        if run >= length:
            return k - length + 1
    return None
