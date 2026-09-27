"""Whole-event analysis: any number of lengths, one clip, one camera.

    from core.event import analyze_event
    event = analyze_event(clip, options, focus="above")     # or "underwater"
    event.metrics, event.key_frames, event.notes, event.details

The swimmer is tracked once over the whole clip. The track is cut into
lengths where the direction of travel flips (the walls). Then, for every
length, the existing phase modules run on just that stretch of the clip:

  dive        on the first length, when the clip shows the start
  turn        around every wall between two lengths (far wall and near wall)
  underwater  from the entry or the push-off to the breakout, every length
  swim        from the breakout to the wall, every length

The per-length results are kept as `details` (named underwater_L3, swim_L3,
turn_L3 = the turn at the end of length 3 ...) and the numbers the coach
compares across lengths are collected into one Result: splits, speed,
strokes and tempo by length, underwater distance by length, turn times.

Which camera the clip comes from decides what is trustworthy. An above-water
camera gives strokes, tempo, splits and turn times; an underwater camera gives
kicks, underwater distance, breakout and push-off. The two event modules
(modules/event_above, modules/event_underwater) pick their metric keys from
the same EventAnalysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.analysis import prepare
from core.base import CACHE_KEYS
from core.signal import crossing_time
from core.types import Result
from core.video import Clip

BODY_AREA_FRACTION = 0.0004


@dataclass
class EventAnalysis:
    metrics: dict[str, Any] = field(default_factory=dict)
    key_frames: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    details: list[Result] = field(default_factory=list)


def _pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or not first:
        return None
    return round((last - first) / first * 100.0, 1)


def _clean(values: list[Any]) -> list[Any] | None:
    """A by-length list, or None when nothing in it was measured."""
    return values if any(v is not None for v in values) else None


def _mean(values: list[Any]) -> float | None:
    good = [float(v) for v in values if v is not None]
    return round(sum(good) / len(good), 2) if good else None


def analyze_event(clip: Clip, options: dict[str, Any], focus: str = "above") -> EventAnalysis:
    from modules.dive.module import DiveModule
    from modules.swim.module import SwimModule, find_lengths
    from modules.turn.module import TurnModule
    from modules.underwater.module import UnderwaterModule

    for key in CACHE_KEYS:
        options.setdefault(key, {})
    out = EventAnalysis()
    pool_m = float(options.get("pool_length_m", 22.86))
    frame_area = clip.width * clip.height

    # 1. one track over the whole clip; fix the outbound direction so every
    #    per-length window maps pixels to metres the same way
    prep = prepare(clip, options, phase="lap")
    if prep.direction != 0 and "swim_direction" not in options:
        options["swim_direction"] = "right" if prep.direction > 0 else "left"
        prep = prepare(clip, options, phase="lap")
    out.notes.extend(prep.notes)
    tr = prep.track
    fps = tr.fps
    n = tr.times.size
    outbound = prep.direction or 1

    seen = np.where(tr.found & (tr.area >= BODY_AREA_FRACTION * frame_area))[0]
    if seen.size < int(fps):
        out.notes.append("event: swimmer not seen long enough to find any lengths")
        return out

    # 2. lengths at direction flips
    lengths = find_lengths(prep, int(seen[0]), int(seen[-1]) + 1)
    count = len(lengths)
    times = tr.times
    flip_t = [float(times[min(n - 1, e)]) for _, e, _ in lengths]
    start_seen_t = float(times[lengths[0][0]])

    # 3. the dive, on the first length only (the full-frame track is limited to it)
    dive: Result | None = None
    from core.track import track_swimmer

    full_opts = dict(options)
    full_opts["full_frame"] = True
    full_opts.pop("_progress", None)
    first_end = min(clip.duration_s, flip_t[0] + 1.0) if count > 1 else None
    options["_tracks_full"].setdefault(id(clip), track_swimmer(clip, full_opts, 0.0, first_end))
    dive = DiveModule().run(clip, options)
    entry_t = dive.metrics.get("entry_time_s")
    has_dive = entry_t is not None
    if has_dive:
        out.details.append(dive)

    # 4. turns at every wall between two lengths
    turns: list[Result | None] = []
    wall_t: list[float | None] = []
    leave_t: list[float | None] = []
    for i in range(count - 1):
        s, e, d = lengths[i]
        s2, e2, d2 = lengths[i + 1]
        mid_a = float(times[(s + e) // 2])
        mid_b = float(times[min(n - 1, (s2 + e2) // 2)])
        opts = dict(options)
        opts["window_start_s"] = mid_a
        opts["window_end_s"] = mid_b
        opts["turn_wall"] = "far" if d == outbound else "near"
        turn = TurnModule().run(clip, opts)
        turn.module = f"turn_L{i + 1}"
        turns.append(turn)
        out.details.append(turn)
        wall = turn.key_frames.get("wall")
        leave = turn.key_frames.get("wall_leave")
        wall_t.append(clip.frame_to_time(int(wall)) if wall is not None else flip_t[i])
        leave_t.append(clip.frame_to_time(int(leave)) if leave is not None else flip_t[i])
        if wall is None:
            out.notes.append(f"event: turn {i + 1} not timed; the wall moment was taken from the direction change")

    # 5. start and finish
    signal_start = float(options.get("start_s", 0.0) or 0.0)
    if signal_start > 0:
        start_t = signal_start
        start_origin = "start signal"
    elif has_dive and dive.metrics.get("movement_time_s") is not None:
        start_t = float(dive.metrics["movement_time_s"])
        start_origin = "first movement on the block"
    else:
        start_t = start_seen_t
        start_origin = "first sight of the swimmer"
        out.notes.append("event: no start signal given; the event time starts at the first sight of the swimmer")
    out.key_frames["start"] = tr.index_at(start_t)
    s_last, e_last, d_last = lengths[-1]
    target = pool_m - 0.3 if d_last == outbound else 0.3
    finish_t = crossing_time(times[s_last:e_last], prep.x_m[s_last:e_last], target, rising=(d_last == outbound))
    if finish_t is None:
        last_seen = int(seen[-1])
        finish_t = float(times[last_seen])
        out.notes.append("event: the final wall touch was not seen; the event time ends at the last sight of the swimmer")
    out.key_frames["finish"] = tr.index_at(finish_t)
    for i, t in enumerate(wall_t):
        out.key_frames[f"wall_{i + 1}"] = tr.index_at(t)

    # 6. underwater and swim on every length
    splits: list[float | None] = []
    cumulative: list[float | None] = []
    speeds: list[float | None] = []
    strokes: list[Any] = []
    tempo: list[Any] = []
    dps: list[Any] = []
    swim_speed: list[Any] = []
    uw_dist: list[Any] = []
    uw_time: list[Any] = []
    kicks: list[Any] = []
    breakout_speed: list[Any] = []
    breakout_slow: list[Any] = []
    peak_uw: list[Any] = []
    absolute = prep.cal.wall_x is not None or prep.cal.method in ("points", "profile", "walls")
    for i, (s, e, d) in enumerate(lengths):
        length_start = start_t if i == 0 else leave_t[i - 1]
        length_end = finish_t if i == count - 1 else wall_t[i]
        if length_end is None or length_start is None or length_end <= length_start:
            for bucket in (splits, cumulative, speeds, strokes, tempo, dps, swim_speed, uw_dist, uw_time, kicks,
                           breakout_speed, breakout_slow, peak_uw):
                bucket.append(None)
            continue
        split = round(length_end - length_start, 2)
        splits.append(split)
        cumulative.append(round(length_end - start_t, 2))
        speeds.append(round(pool_m / split, 2) if split > 0 else None)

        uw_opts = dict(options)
        uw_opts["window_start_s"] = float(entry_t) if (i == 0 and has_dive) else float(length_start)
        uw_opts["window_end_s"] = max(uw_opts["window_start_s"] + 0.5, float(length_end) - 0.3)
        if absolute:
            uw_opts["from_wall_m"] = 0.0 if d == outbound else pool_m
        under = UnderwaterModule().run(clip, uw_opts)
        under.module = f"underwater_L{i + 1}"
        out.details.append(under)
        uw_dist.append(under.metrics.get("underwater_distance_m"))
        uw_time.append(under.metrics.get("underwater_time_s"))
        kicks.append(under.metrics.get("kick_count"))
        breakout_speed.append(under.metrics.get("breakout_speed_mps"))
        breakout_slow.append(under.metrics.get("breakout_slowdown_pct"))
        peak_uw.append(under.metrics.get("peak_underwater_speed_mps"))

        swim_opts = dict(options)
        breakout = under.key_frames.get("breakout")
        swim_opts["window_start_s"] = clip.frame_to_time(int(breakout)) if breakout is not None else uw_opts["window_start_s"]
        swim_opts["window_end_s"] = float(length_end)
        swim = SwimModule().run(clip, swim_opts)
        swim.module = f"swim_L{i + 1}"
        out.details.append(swim)
        strokes.append(swim.metrics.get("stroke_count") or None)
        tempo.append(swim.metrics.get("stroke_rate_spm"))
        dps.append(swim.metrics.get("distance_per_stroke_m"))
        swim_speed.append(swim.metrics.get("avg_speed_mps"))

    # 7. the numbers to compare across lengths
    m = out.metrics
    m["lengths"] = count
    m["distance_m"] = round(count * pool_m, 1)
    m["event_time_s"] = round(finish_t - start_t, 2)
    m["avg_speed_mps"] = round(count * pool_m / (finish_t - start_t), 2) if finish_t > start_t else None
    m["split_by_length_s"] = _clean(splits)
    m["cumulative_split_s"] = _clean(cumulative)
    m["speed_by_length_mps"] = _clean(speeds)
    # the first length has the dive in it, so the change is measured from the second length when there is one
    ref = 1 if count >= 3 else 0
    m["speed_change_pct"] = _pct_change(speeds[ref], speeds[-1]) if count >= 2 else None
    m["strokes_by_length"] = _clean(strokes)
    m["tempo_by_length_spm"] = _clean(tempo)
    m["distance_per_stroke_by_length_m"] = _clean(dps)
    m["surface_speed_by_length_mps"] = _clean(swim_speed)
    m["tempo_change_pct"] = _pct_change(tempo[ref], tempo[-1]) if count >= 2 else None
    m["strokes_change_pct"] = _pct_change(strokes[ref], strokes[-1]) if count >= 2 else None
    m["underwater_distance_by_length_m"] = _clean(uw_dist)
    m["underwater_time_by_length_s"] = _clean(uw_time)
    m["underwater_change_pct"] = _pct_change(uw_dist[ref], uw_dist[-1]) if count >= 2 else None
    m["kicks_by_length"] = _clean(kicks)
    m["breakout_speed_by_length_mps"] = _clean(breakout_speed)
    m["breakout_slowdown_by_length_pct"] = _clean(breakout_slow)
    m["peak_underwater_speed_by_length_mps"] = _clean(peak_uw)
    m["turn_count"] = len(turns)
    m["turn_time_by_turn_s"] = _clean([t.metrics.get("turn_time_s") for t in turns]) if turns else None
    m["turn_time_avg_s"] = _mean([t.metrics.get("turn_time_s") for t in turns]) if turns else None
    m["wall_contact_by_turn_s"] = _clean([t.metrics.get("wall_contact_s") for t in turns]) if turns else None
    m["push_off_speed_by_turn_mps"] = _clean([t.metrics.get("push_off_speed_mps") for t in turns]) if turns else None
    m["turn_type"] = next((t.metrics.get("turn_type") for t in turns if t.metrics.get("turn_type")), None)
    for key in ("reaction_time_s", "flight_time_s", "entry_distance_m", "entry_angle_deg", "entry_depth_m"):
        m[key] = dive.metrics.get(key) if has_dive else None

    # details in reading order: dive, then per length underwater, swim, turn
    ordered: list[Result] = [d for d in out.details if d.module == "dive"]
    for i in range(count):
        for prefix in ("underwater", "swim", "turn"):
            ordered.extend(d for d in out.details if d.module == f"{prefix}_L{i + 1}")
    out.details = ordered

    first_split_note = "" if signal_start > 0 else f" (timed from the {start_origin})"
    out.notes.append(f"event: {count} length{'s' if count != 1 else ''} found, {len(turns)} turn{'s' if len(turns) != 1 else ''}; "
                     f"event time {m['event_time_s']:.2f} s{first_split_note}")
    if prep.cal.confidence == "low":
        out.notes.append("event: distances and speeds use a guessed scale; calibrate the camera for real numbers")
    if focus == "underwater":
        out.notes.append("event: strokes and tempo from an underwater camera are rough; the above-water event module is better for those")
    else:
        out.notes.append("event: kick counts from above the water are rough; the underwater event module is better for those")
    return out
