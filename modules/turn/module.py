"""Turn module: approach to the wall, the turn itself, and the push-off.

Metrics (from the team meeting):
  turn_type          "flip" or "open" (set -o turn_type=flip to force it)
  turn_in_time_s     time from the 5 m mark to the wall (closest approach)
  turn_out_time_s    time from the wall back out to the 5 m mark
  turn_time_s        turn_in + turn_out (the 5 m in, 5 m out turn time coaches use)
  wall_contact_s     time the swimmer is stopped at the wall
  push_off_speed_mps fastest speed in the first second after leaving the wall
  turn_zone_m        the distance used on each side of the wall (5 m, or less when the clip is short)

Key frames: "turn_in", "wall", "wall_leave", "turn_out".

How it works: the swimmer's distance from the wall is followed along the
lane. The wall moment is the closest approach where the direction of travel
flips. Contact lasts while the swimmer is within about a body length of the
wall and hardly moving. The turn zone is 5 m in and 5 m out when the
calibration is good and the clip shows that much; otherwise the largest
distance visible on both sides is used and reported.

Flip or open: in a flip turn the body tucks (the blob gets short and tall)
and the head goes under before the wall; in an open turn the swimmer stays
long and touches with the hands. The blob's shape in the last half second
before the wall decides, unless turn_type is given.

Extra options this module reads:
  turn_type=flip|open
  turn_wall=auto|far|near  which wall the turn is at (auto tries the far wall, then the near one)
  turn_zone_m=5           metres each side of the wall
  calib_points=x1:m1,x2:m2  (or calib_profile=name) for real distances
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.analysis import Prepared, prepare
from core.base import AnalysisModule
from core.signal import crossing_time, fill_gaps, moving_average
from core.types import Result
from core.video import Clip


def find_wall_moment(prep: Prepared, pool_m: float, wall: str = "far") -> tuple[int, float] | None:
    """(track position of closest approach, wall position in metres) or None.

    wall="far" looks for the turn at the far wall (the swimmer goes out, comes
    back); wall="near" for the turn at the start wall (comes in, goes out
    again), which a whole-event clip has every second length.
    """
    tr = prep.track
    fps = tr.fps
    x = moving_average(np.where(tr.found, prep.x_m, np.nan), int(round(0.3 * fps)))
    known = np.isfinite(x)
    x = fill_gaps(x)
    if not known.any():
        return None
    if wall == "near":
        # mirror the lane so the near wall looks like the far wall
        found = find_wall_moment(_mirrored(prep, x), pool_m, "far")
        return None if found is None else (found[0], float(x[found[0]]))
    # the turn wall is the far wall for the outbound direction: x_m grows toward it
    vx = np.gradient(x, tr.times)
    sign = np.sign(moving_average(vx, int(round(0.5 * fps))))
    # find the flip from + to - (outbound then inbound) with movement on both sides
    best = None
    for k in range(int(0.5 * fps), tr.times.size - int(0.5 * fps)):
        before = sign[max(0, k - int(1.0 * fps)) : k]
        after = sign[k : k + int(1.0 * fps)]
        seen = known[max(0, k - int(1.0 * fps)) : k + int(1.0 * fps)]
        if seen.mean() < 0.6:
            continue
        if before.size and after.size and np.mean(before > 0) > 0.6 and np.mean(after < 0) > 0.6:
            # real turn: the swimmer comes back at least 2 m within 2 s
            back = x[k : k + int(2.0 * fps)]
            if back.size < 2 or float(x[k] - np.nanmin(back)) < 2.0:
                continue
            score = float(x[k])
            if best is None or score > best[1]:
                best = (k, score)
    if best is None:
        return None
    k = best[0]
    lo, hi = max(0, k - int(0.7 * fps)), min(tr.times.size, k + int(0.7 * fps))
    k = lo + int(np.nanargmax(x[lo:hi]))
    return k, float(x[k])


class TurnModule(AnalysisModule):
    name = "turn"
    description = "Flip and open turns"
    owner = ""
    metric_keys = (
        "turn_type",
        "turn_in_time_s",
        "turn_out_time_s",
        "turn_time_s",
        "wall_contact_s",
        "push_off_speed_mps",
        "turn_zone_m",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        window_start = float(options.get("window_start_s", 0.0))
        window_end = options.get("window_end_s")
        prep = prepare(clip, options, phase="turn", start_s=window_start, end_s=window_end)
        result.notes.extend(prep.notes)
        tr = prep.track
        fps = tr.fps
        n = tr.times.size
        k_all = np.arange(n)
        pool_m = float(options.get("pool_length_m", 22.86))

        wanted = str(options.get("turn_wall", "auto")).lower()
        walls = ["far", "near"] if wanted not in ("far", "near") else [wanted]
        found = None
        wall_side = "far"
        for wall_side in walls:
            found = find_wall_moment(prep, pool_m, wall_side)
            if found is not None:
                break
        if found is None:
            result.notes.append("turn: no turn seen (the swimmer never reaches the wall and comes back)")
            return result
        wall_k, wall_m = found
        wall_t = float(tr.times[wall_k])
        result.key_frames["wall"] = int(tr.indexes[wall_k])
        x = fill_gaps(moving_average(np.where(tr.found, prep.x_m, np.nan), int(round(0.2 * fps))))
        # distance from the wall along the lane, positive on both sides
        if wall_side == "near":
            wall_pos = 0.0 if abs(wall_m) < 3.0 else wall_m
            dist = x - wall_pos
        else:
            wall_pos = pool_m if abs(wall_m - pool_m) < 3.0 else wall_m
            dist = wall_pos - x
        if prep.cal.confidence == "low":
            result.notes.append("turn: distances use a guessed scale; calibrate the camera to get real 5 m in / 5 m out times")

        zone = float(options.get("turn_zone_m", 5.0))
        seen_in = float(np.nanmax(dist[: wall_k + 1])) if wall_k > 0 else 0.0
        seen_out = float(np.nanmax(dist[wall_k:])) if wall_k < n - 1 else 0.0
        usable = min(zone, seen_in, seen_out)
        if usable < 1.0:
            result.notes.append("turn: the clip does not show enough of the approach and push-off to time the turn")
        else:
            if usable < zone - 0.05:
                result.notes.append(f"turn: only {usable:.1f} m visible on both sides of the wall, so that zone is used instead of {zone:g} m")
            result.metrics["turn_zone_m"] = round(usable, 2)
            t_in = _last_crossing(tr.times[: wall_k + 1], dist[: wall_k + 1], usable, falling=True)
            t_out = crossing_time(tr.times[wall_k:], dist[wall_k:], usable, rising=True)
            if t_in is not None and t_out is not None:
                result.metrics["turn_in_time_s"] = round(wall_t - t_in, 2)
                result.metrics["turn_out_time_s"] = round(t_out - wall_t, 2)
                result.metrics["turn_time_s"] = round(t_out - t_in, 2)
                result.key_frames["turn_in"] = tr.index_at(t_in)
                result.key_frames["turn_out"] = tr.index_at(t_out)

        # wall contact: within a body length of the wall and hardly moving
        near = np.abs(dist) < 1.2
        slow = np.abs(np.nan_to_num(prep.speed)) < 0.6
        stopped = near & slow
        a = wall_k
        while a > 0 and stopped[a - 1]:
            a -= 1
        b = wall_k
        while b < n - 1 and stopped[b + 1]:
            b += 1
        contact = float(tr.times[b] - tr.times[a])
        result.metrics["wall_contact_s"] = round(contact, 2)
        result.key_frames["wall_leave"] = int(tr.indexes[b])

        # push-off speed: fastest in the first second after leaving the wall
        after = (k_all > b) & (k_all <= b + int(round(1.0 * fps))) & np.isfinite(prep.speed)
        if after.any():
            result.metrics["push_off_speed_mps"] = round(float(np.abs(prep.speed[after]).max()), 2)

        # flip or open
        forced = str(options.get("turn_type", "")).lower()
        if forced in ("flip", "open"):
            result.metrics["turn_type"] = forced
        else:
            before = (k_all >= wall_k - int(round(0.5 * fps))) & (k_all < wall_k) & tr.found
            if before.sum() >= 3:
                w = tr.right[before] - tr.left[before]
                h = tr.bottom[before] - tr.top[before]
                tall = float(np.median(h / np.maximum(1.0, w)))
                if tall > 0.8 and contact < 0.6:
                    result.metrics["turn_type"] = "flip"
                else:
                    result.metrics["turn_type"] = "open"
                result.notes.append("turn: type guessed from the body shape at the wall; set -o turn_type=flip or open to override")
        return result


def _mirrored(prep: Prepared, x: np.ndarray) -> Prepared:
    """A shallow copy of prep whose x_m runs the other way (near wall becomes far wall)."""
    import copy

    flipped = copy.copy(prep)
    flipped.x_m = -x
    return flipped


def _last_crossing(times: np.ndarray, values: np.ndarray, level: float, falling: bool = True) -> float | None:
    out = None
    for i in range(1, values.size):
        a, b = values[i - 1], values[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (falling and a > level >= b) or (not falling and a < level <= b):
            frac = (level - a) / (b - a) if b != a else 0.0
            out = float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return out
