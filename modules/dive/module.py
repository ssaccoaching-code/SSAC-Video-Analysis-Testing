"""Dive module: from the start signal to the swimmer entering the water.

Metrics (from the team meeting):
  reaction_time_s      time from the start signal to the feet leaving the block
                       (needs the start signal: --start, or a strobe flash in the clip)
  flight_time_s        time from leaving the block to hands entering the water
  entry_distance_m     distance from the wall to the entry point
  entry_angle_deg      angle of the body relative to the water surface at entry
  entry_path_angle_deg angle of the flight path (centre of the body) just before entry
  entry_depth_m        how deep the swimmer goes right after entry (underwater camera only)
  movement_time_s      time in the clip when the swimmer starts the dive (handy when no start signal)
  entry_time_s         time in the clip of the entry

Key frames: "movement", "leave_block", "entry", "deepest".

How it works: the shared tracker (core/track.py) follows the swimmer over the
whole picture. The flight is the fastest horizontal movement in the clip.
Entry is where the body starts dropping fastest into the water after the
flight; on an underwater camera it is the first frame the body is seen below
the surface. Walking back from the flight, the movement starts
where the body first becomes a moving blob. The body angle comes from the
blob's long axis just before entry; the path angle from the trajectory.

Extra options this module reads:
  view=underwater|above     (auto-detected if not set)
  calib_points=x1:m1,x2:m2  (or calib_profile=name) for real distances
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.analysis import Prepared, prepare
from core.base import AnalysisModule
from core.signal import derivative, moving_average
from core.types import Result
from core.video import Clip

BODY_AREA_FRACTION = 0.0015   # of the frame: a person-sized moving blob


def detect_start_flash(clip: Clip, before_s: float, step: int = 1) -> float | None:
    """A strobe start signal shows as a one-frame brightness jump. Returns its time or None."""
    import cv2

    prev = None
    best_t, best_jump = None, 0.0
    for index, t, frame in clip.frames(end_s=before_s, step=step):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        level = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        if prev is not None:
            jump = level - prev
            if jump > best_jump:
                best_jump, best_t = jump, t
        prev = level
    return best_t if best_jump > 25.0 else None


def find_dive_events(prep: Prepared, frame_area: float) -> dict[str, int] | None:
    """Frame positions (into the track arrays) of movement, leave_block and entry.

    The result also carries "direction": +1 when the dive goes toward larger x.
    """
    tr = prep.track
    fps = tr.fps
    n = tr.times.size
    k_all = np.arange(n)
    body = tr.found & (tr.area >= BODY_AREA_FRACTION * frame_area)
    if body.sum() < 3:
        return None

    # the flight: fastest sustained horizontal movement of a body-sized blob
    # that ends in a splash (the biggest burst of white foam in the clip).
    # Both directions are tried; other people walking on deck must not win.
    vx_raw = derivative(moving_average(np.where(body, prep.x_px, np.nan), 3), tr.times)
    vx_raw = np.where(body, vx_raw, np.nan)
    hold = max(3, int(round(0.2 * fps)))
    white = np.where(tr.found, tr.white, 0.0)
    ahead = int(round(1.5 * fps))
    white_ahead = np.array([white[k : k + ahead].max() for k in range(n)])
    big_splash = white_ahead >= 0.5 * white.max() if white.max() > 0 else np.ones(n, dtype=bool)
    best = None
    for direction in (1, -1):
        vx_s = moving_average(vx_raw * direction, max(3, int(round(0.1 * fps))))
        vx_min = np.array([np.nanmin(vx_s[k : k + hold]) if np.isfinite(vx_s[k : k + hold]).all() else np.nan for k in range(n)])
        score = np.where(big_splash, vx_min, np.nan)
        if not np.isfinite(score).any():
            score = vx_min
        if not np.isfinite(score).any():
            continue
        k = int(np.nanargmax(score))
        if best is None or score[k] > best[0]:
            best = (float(score[k]), k, direction, vx_s)
    if best is None or best[0] <= 0:
        return None
    _, peak_k, direction, vx_s = best
    peak_v = float(vx_s[peak_k])

    # leaving the block: walk back from the peak while the speed stays high
    leave_k = peak_k
    while leave_k > 0 and np.isfinite(vx_s[leave_k - 1]) and vx_s[leave_k - 1] >= 0.35 * peak_v:
        leave_k -= 1

    # entry: on an underwater camera, the first frame the body is seen below
    # the surface. From above, the blob drops fastest as the body follows the
    # hands into the water; entry is where that drop gets going (60% of peak).
    horizon = min(n - 1, peak_k + int(round(1.5 * fps)))
    window = (k_all >= leave_k) & (k_all <= horizon)
    entry_k = None
    if prep.view == "underwater" and prep.surface_y is not None:
        below = window & body & (tr.top > prep.surface_y)
        idx = np.where(below)[0]
        entry_k = int(idx[0]) if idx.size else None
    if entry_k is None:
        vy = derivative(moving_average(prep.y_px, max(3, int(round(0.1 * fps)))), tr.times)
        vy = np.where(window & (k_all > leave_k), vy, np.nan)
        if np.isfinite(vy).any():
            drop_k = int(np.nanargmax(vy))
            level = 0.85 * float(vy[drop_k])
            entry_k = drop_k
            while entry_k > leave_k + 1 and np.isfinite(vy[entry_k - 1]) and vy[entry_k - 1] >= level:
                entry_k -= 1
    if entry_k is None:
        entry_k = min(n - 1, peak_k + int(round(0.3 * fps)))

    # movement: walk back from leave_block while the blob stays body-sized;
    # before the swimmer moves there is no blob (or a much smaller one)
    near = (k_all >= max(0, leave_k - int(round(0.1 * fps)))) & (k_all <= min(n - 1, leave_k + int(round(0.1 * fps)))) & body
    ref_area = float(np.median(tr.area[near])) if near.any() else float(tr.area[leave_k])
    move_k = leave_k
    k = leave_k - 1
    limit = max(0, leave_k - int(round(0.8 * fps)))
    while k >= limit and body[k] and tr.area[k] >= 0.4 * ref_area:
        move_k = k
        k -= 1
    return {"movement": move_k, "leave_block": leave_k, "entry": entry_k, "peak": peak_k, "direction": direction}


class DiveModule(AnalysisModule):
    name = "dive"
    description = "Start reaction, flight, and water entry"
    owner = ""
    metric_keys = (
        "reaction_time_s",
        "flight_time_s",
        "entry_distance_m",
        "entry_angle_deg",
        "entry_path_angle_deg",
        "entry_depth_m",
        "movement_time_s",
        "entry_time_s",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        window_end = options.get("window_end_s")
        in_lap = "window_start_s" in options or window_end is not None
        prep = prepare(clip, options, phase="dive", start_s=float(options.get("window_start_s", 0.0)), end_s=window_end,
                       full_frame=not in_lap)
        result.notes.extend(prep.notes)
        tr = prep.track
        fps = tr.fps

        events = find_dive_events(prep, clip.width * clip.height)
        if events is None:
            result.notes.append("dive: could not see a dive; is the block in the picture and the swimmer moving?")
            return result
        if events["direction"] != prep.direction:
            # the dive goes the other way than the rest of the motion in the clip
            forced = dict(options)
            forced["swim_direction"] = "right" if events["direction"] > 0 else "left"
            prep = prepare(clip, forced, phase="dive", start_s=float(options.get("window_start_s", 0.0)), end_s=window_end,
                           full_frame=not in_lap)
            tr = prep.track
        move_k, leave_k, entry_k = events["movement"], events["leave_block"], events["entry"]
        entry_t = float(tr.times[entry_k])
        flight_check = entry_t - float(tr.times[leave_k])
        lead_px = tr.right[entry_k] if events["direction"] > 0 else tr.left[entry_k]
        lead_m = float(prep.cal.x_to_m(lead_px)) if np.isfinite(lead_px) else float("nan")
        too_far = prep.cal.confidence == "high" and np.isfinite(lead_m) and not (0.3 < lead_m < 7.0)
        if not (0.15 < flight_check < 1.2) or too_far:
            result.notes.append("dive: no dive seen in this clip (nothing flies off a block into a splash near the wall)")
            return result
        if prep.cal.method == "walls" and np.isfinite(lead_m) and not (0.3 < lead_m < 7.0):
            # a dive camera rarely shows both walls; that guess was wrong, so use the plain default scale
            from core.calibrate import Calibration, DEFAULT_SPAN_M

            span = DEFAULT_SPAN_M["dive"]
            points = [(0.0, 0.0), (float(clip.width), span)] if events["direction"] > 0 else [(float(clip.width), 0.0), (0.0, span)]
            prep.cal = Calibration(points, lane_y=prep.cal.lane_y, method="default", confidence="low")
            result.notes.append("dive: the automatic wall calibration did not fit this clip; distances use a guessed scale. "
                                "Save a calibration for the dive camera to get real numbers.")
        result.metrics["movement_time_s"] = round(float(tr.times[move_k]), 3)
        result.metrics["entry_time_s"] = round(entry_t, 3)
        result.key_frames["movement"] = int(tr.indexes[move_k])
        result.key_frames["leave_block"] = int(tr.indexes[leave_k])
        result.key_frames["entry"] = int(tr.indexes[entry_k])

        flight_s = entry_t - float(tr.times[leave_k])
        if 0.05 < flight_s < 2.0:
            result.metrics["flight_time_s"] = round(flight_s, 3)
        else:
            result.notes.append("dive: flight time looked wrong and was left out")

        # reaction time needs a start signal
        start_s = float(options.get("start_s", 0.0) or 0.0)
        if start_s <= 0:
            flash = detect_start_flash(clip, before_s=float(tr.times[move_k]))
            if flash is not None:
                start_s = flash
                result.notes.append(f"dive: start flash seen at {flash:.2f} s")
        if start_s > 0 and tr.times[leave_k] > start_s:
            # official reaction time: start signal to the feet leaving the block
            result.metrics["reaction_time_s"] = round(float(tr.times[leave_k]) - start_s, 3)
            result.key_frames["start"] = clip.time_to_frame(start_s)
        else:
            result.notes.append("dive: no start signal (use --start or a strobe in the clip), so reaction time is blank")

        # angles just before entry: the flight path of the body centre, and the body axis
        k_all = np.arange(tr.times.size)
        k0 = max(leave_k, entry_k - int(round(0.2 * fps)))
        if entry_k > k0 and np.isfinite(prep.x_px[k0]) and np.isfinite(prep.x_px[entry_k]):
            dx = (prep.x_px[entry_k] - prep.x_px[k0]) * prep.direction
            dy = prep.y_px[entry_k] - prep.y_px[k0]
            if abs(dx) > 1e-6:
                result.metrics["entry_path_angle_deg"] = round(float(np.degrees(np.arctan2(dy, abs(dx)))), 1)
        pre = (k_all >= max(k0, entry_k - 3)) & (k_all < entry_k) & tr.found
        body_angles = tr.angle[pre]
        body_angles = body_angles[np.isfinite(body_angles)]
        if body_angles.size:
            result.metrics["entry_angle_deg"] = round(float(np.median(np.abs(body_angles))), 1)
        elif result.metrics["entry_path_angle_deg"] is not None:
            result.metrics["entry_angle_deg"] = result.metrics["entry_path_angle_deg"]
        result.notes.append("dive: entry_angle_deg is the body line seen by the camera; entry_path_angle_deg is the flight path of the body centre over the last 0.2 s")

        # entry distance from the wall: front of the body at entry
        lead = tr.right[entry_k] if prep.direction > 0 else tr.left[entry_k]
        if np.isfinite(lead):
            metres = float(prep.cal.x_to_m(lead))
            result.metrics["entry_distance_m"] = round(metres, 2)
            if prep.cal.confidence == "low":
                result.notes.append("dive: entry distance uses a guessed scale; calibrate the camera for a real number")

        # depth: only when we can see below the surface
        if prep.view == "underwater" and prep.surface_y is not None:
            look = (k_all > entry_k) & (k_all <= entry_k + int(round(2.0 * fps))) & tr.found
            if look.any():
                deep_k = int(np.argmax(np.where(look, tr.bottom, -np.inf)))
                depth_px = float(tr.bottom[deep_k] - prep.surface_y)
                depth_m = depth_px * prep.cal.m_per_px(prep.x_px[deep_k])
                if np.isfinite(depth_m) and depth_m > 0:
                    result.metrics["entry_depth_m"] = round(depth_m, 2)
                    result.key_frames["deepest"] = int(tr.indexes[deep_k])
        else:
            result.notes.append("dive: entry depth needs the underwater camera")
        return result
