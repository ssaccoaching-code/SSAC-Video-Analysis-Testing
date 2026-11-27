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
  swimmer_height_m=1.75     used for the fallback scale when the camera is not calibrated

Without a calibration the module does not trust the frame width. It puts 0 m
at the block edge where the feet leave and takes the scale from the swimmer's
own stretched body in flight (about 1.28 x height, fingertips to toes). That is
a rough number (low confidence) but it is anchored to things in the picture.
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
DEFAULT_HEIGHT_M = 1.75       # when the swimmer's height is not given
STRETCH_FACTOR = 1.28         # fingertips to toes in streamline, as a multiple of standing height


def self_calibration(prep: Prepared, events: dict[str, int], swimmer_height_m: float) -> "Calibration | None":
    """A calibration from the dive itself, for cameras that were never calibrated.

    0 m is the block edge: the trailing edge of the body (the feet) on the frame
    the swimmer leaves the block. Metres per pixel come from the longest the
    body gets in flight, which is the swimmer stretched out, about 1.28 x height.
    Both are read in the swimmer's own lane, so perspective and camera angle
    cancel out for the short distance a dive covers.
    """
    from core.calibrate import Calibration

    tr = prep.track
    direction = events["direction"]
    leave_k, entry_k = events["leave_block"], events["entry"]
    n = tr.times.size

    # the stretched body in flight: widest clean blob between take-off and entry
    flight = [k for k in range(leave_k + 1, entry_k + 1) if tr.found[k] and np.isfinite(tr.left[k]) and np.isfinite(tr.right[k])]
    if len(flight) < 3:
        return None
    whites = np.array([tr.white[k] for k in flight])
    clean = [k for k, w in zip(flight, whites) if w <= max(0.3 * whites.max(), 1.0)] or flight
    extents = np.array([tr.right[k] - tr.left[k] for k in clean])
    body_px = float(np.percentile(extents, 90))
    if body_px < 20:
        return None
    m_per_px = STRETCH_FACTOR * swimmer_height_m / body_px

    # the feet on the block edge, just as they leave
    trailing = tr.right if direction < 0 else tr.left
    leading = tr.left if direction < 0 else tr.right
    near = [k for k in range(leave_k, min(n, leave_k + 3)) if tr.found[k] and np.isfinite(trailing[k])]
    if not near:
        return None
    wall_x = float(max(trailing[k] for k in near) if direction < 0 else min(trailing[k] for k in near))
    merged = False
    if float(tr.right[near[0]] - tr.left[near[0]]) > 1.1 * body_px:
        # the blob at take-off also covers people or railings behind the block, so its
        # trailing edge is not the feet. Take the first clean flight frame and move the
        # feet back to the block at the flight speed.
        merged = True
        clean_k = [k for k in flight if tr.right[k] - tr.left[k] <= 1.1 * body_px]
        if len(clean_k) >= 2:
            c = clean_k[0]
            steps = np.abs(np.diff([leading[k] for k in clean_k[:6]]))
            vx = float(np.median(steps)) if steps.size else 0.0
            wall_x = float(trailing[c] - direction * vx * (c - leave_k))
    far_x = wall_x + direction * 1000.0
    cal = Calibration([(wall_x, 0.0), (far_x, 1000.0 * m_per_px)], lane_y=prep.cal.lane_y, surface_y=prep.cal.surface_y,
                      wall_x=wall_x, method="dive_self", confidence="low")
    cal.notes.append(f"dive: no camera calibration, so 0 m is the block edge where the feet leave (x={wall_x:.0f} px) and the "
                     f"scale assumes a {swimmer_height_m:.2f} m swimmer stretched to {STRETCH_FACTOR * swimmer_height_m:.2f} m "
                     f"({body_px:.0f} px in flight). Rough; set swimmer_height_m or calibrate the camera for a real number.")
    if merged:
        cal.notes.append("dive: at take-off the swimmer overlapped people or railings behind the block, so the block edge was "
                         "estimated from the flight; the entry distance could be off by a few tenths of a metre.")
    return cal


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

    # the first sustained burst of foam after take-off is the hands hitting the water;
    # the entry can never be later than that (the big splash comes after it)
    # (the blob often overlaps white railings and caps behind the block at take-off,
    # so the search starts after the white has gone quiet in flight)
    splash_k = None
    span = white[leave_k : horizon + 1]
    if span.size >= 6 and np.isfinite(span).any():
        base, top = float(np.nanmin(span)), float(np.nanmax(span))
        quiet_level = base + 0.1 * (top - base)
        quiet_k = None
        for k in range(leave_k + 1, horizon - 1):
            if np.all(white[k : k + 3] <= quiet_level):
                quiet_k = k
                break
        if quiet_k is not None:
            top = float(np.nanmax(white[quiet_k : horizon + 1]))
            level = base + max(0.25 * (top - base), 8.0)
            if top - base >= 20.0:
                for k in range(quiet_k + 3, horizon - 1):
                    if np.all(white[k : k + 3] >= level):
                        splash_k = k
                        break
    if splash_k is not None and splash_k < entry_k:
        entry_k = splash_k

    # a dive is a body going roughly head-first into the water; a person standing or
    # walking on deck is a vertical blob the whole way, and no dive at all
    flight = (k_all > leave_k) & (k_all <= entry_k) & body & np.isfinite(tr.angle)
    if flight.sum() >= 3 and float(np.median(np.abs(tr.angle[flight]))) > 65.0:
        return None

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
        guessed = prep.cal.confidence == "low" or prep.cal.method == "default"
        walls_wrong = prep.cal.method == "walls" and np.isfinite(lead_m) and not (0.3 < lead_m < 7.0)
        if guessed or walls_wrong:
            # the frame-width guess (and a wall guess on a camera that shows one wall) put 0 m at the
            # edge of the picture; a dive camera almost never does. Read wall and scale off the dive itself.
            from core.analysis import recalibrate

            height = float(options.get("swimmer_height_m") or DEFAULT_HEIGHT_M)
            own = self_calibration(prep, events, height)
            if own is not None:
                recalibrate(prep, own)
                result.notes.extend(own.notes)
            elif walls_wrong:
                result.notes.append("dive: the automatic wall calibration did not fit this clip and the dive itself could not be "
                                    "measured; distances use a guessed scale. Save a calibration for the dive camera.")
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
                result.notes.append("dive: entry distance uses a rough scale; calibrate the camera for a real number")

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
