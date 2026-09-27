"""Swim module: the surface swimming between breakout and the wall.

Metrics (from the team meeting):
  stroke_count          strokes taken on the surface (all lengths in the clip)
  stroke_rate_spm       tempo, strokes per minute, over the whole surface swim
  distance_per_stroke_m metres travelled per stroke
  avg_speed_mps         average surface speed
  swim_time_s           time spent swimming on the surface
  splits_s              time at each marker from the start signal (or the clip start), e.g. {"5m": 3.1, "15m": 8.2, "wall": 13.4}
  tempo_by_lap          stroke rate per length, in order
  strokes_by_lap        stroke count per length, in order
  tempo_change_pct      percent change of tempo from the first length to the last (negative = slowed down)

Key frames: "swim_start", "finish", plus one per length ("length_2", ...).

How it works: the surface swim runs from the breakout (or the first frame the
swimmer is seen on the surface) to the wall. Lengths are split where the
swimmer turns around (the direction of travel flips). Each arm entry makes a
burst of foam and a wider blob, so strokes are the peaks of that rhythm
between 0.6 and 2.5 per second. For freestyle and backstroke each arm counts
as one stroke; for butterfly and breaststroke each cycle counts as one.

Extra options this module reads:
  view=underwater|above     (auto-detected if not set)
  calib_points=x1:m1,x2:m2  (or calib_profile=name) for real distances and splits
  markers=5,10,15           split marks in metres from the start wall (default depends on the pool)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.analysis import Prepared, prepare
from core.base import AnalysisModule
from core.signal import count_rhythm, crossing_time, fill_gaps, moving_average
from core.types import Result
from core.video import Clip

DEFAULT_MARKERS = {"SCY": [5.0, 10.0, 15.0], "SCM": [5.0, 10.0, 15.0], "LCM": [15.0, 25.0, 35.0]}


def find_lengths(prep: Prepared, from_k: int, to_k: int) -> list[tuple[int, int, int]]:
    """Split [from_k, to_k] into lengths at direction flips: [(start_k, end_k, direction), ...]."""
    tr = prep.track
    fps = tr.fps
    x = moving_average(prep.x_px, int(round(0.5 * fps)))
    x = fill_gaps(x)
    vx = np.gradient(x, tr.times)
    sign = np.sign(moving_average(vx, int(round(0.5 * fps))))
    lengths: list[tuple[int, int, int]] = []
    k = from_k
    while k < to_k:
        d = int(sign[k]) if sign[k] != 0 else 0
        j = k
        while j < to_k and (sign[j] == d or sign[j] == 0):
            j += 1
        travel = abs(float(x[min(j, x.size - 1)] - x[k])) if j > k else 0.0
        if lengths and d == lengths[-1][2]:
            # same direction as the last length: it is the same length
            s, e, pd = lengths[-1]
            lengths[-1] = (s, j, pd)
        elif d != 0 and j - k >= int(1.5 * fps) and travel >= 0.15 * prep.background.shape[1]:
            lengths.append((k, j, d))
        elif lengths and d != 0:
            # a short blip: glue it to the previous length
            s, e, pd = lengths[-1]
            lengths[-1] = (s, j, pd)
        k = j if j > k else k + 1
    if not lengths:
        lengths = [(from_k, to_k, prep.direction or 1)]
    return lengths


def count_strokes(prep: Prepared, start_k: int, end_k: int) -> tuple[int, list[int]]:
    """Strokes in [start_k, end_k) from the foam and blob-width rhythm."""
    tr = prep.track
    fps = tr.fps
    pick = np.zeros(tr.times.size, dtype=bool)
    pick[start_k:end_k] = True
    pick &= tr.found
    if pick.sum() < int(0.8 * fps):
        return 0, []
    width = np.where(pick, tr.right - tr.left, np.nan)
    foam = np.where(pick, tr.white, np.nan)
    def norm(v: np.ndarray) -> np.ndarray:
        v = fill_gaps(v)
        spread = np.nanpercentile(v, 90) - np.nanpercentile(v, 10)
        return (v - np.nanmedian(v)) / spread if spread > 0 else np.zeros_like(v)
    area = np.where(pick, tr.area, np.nan)
    signal = norm(width) + norm(area) + 0.5 * norm(foam)
    sub = signal[start_k:end_k]
    count, peaks, period = count_rhythm(sub, fps, min_period_s=0.35, max_period_s=1.6)
    return count, [start_k + p for p in peaks]


class SwimModule(AnalysisModule):
    name = "swim"
    description = "Stroke count, tempo, speed and splits"
    owner = ""
    metric_keys = (
        "stroke_count",
        "stroke_rate_spm",
        "distance_per_stroke_m",
        "avg_speed_mps",
        "swim_time_s",
        "splits_s",
        "tempo_by_lap",
        "strokes_by_lap",
        "tempo_change_pct",
    )

    def analyze(self, clip: Clip, options: dict[str, Any]) -> Result:
        result = self.empty_result()
        window_start = float(options.get("window_start_s", 0.0))
        window_end = options.get("window_end_s")
        in_lap = "window_start_s" in options or window_end is not None
        prep = prepare(clip, options, phase="swim", start_s=window_start, end_s=window_end)
        result.notes.extend(prep.notes)
        tr = prep.track
        fps = tr.fps
        n = tr.times.size
        k_all = np.arange(n)
        frame_area = clip.width * clip.height
        pool_m = float(options.get("pool_length_m", 22.86))

        seen = np.where(tr.found & (tr.area >= 0.0004 * frame_area))[0]
        if seen.size < int(fps):
            result.notes.append("swim: swimmer not seen long enough")
            return result

        # surface swim starts at the breakout when the clip has one
        start_k = int(seen[0])
        if not in_lap:
            from modules.underwater.module import find_breakout

            b = find_breakout(prep, int(seen[0]), frame_area)
            if b is not None and tr.times[b] - tr.times[seen[0]] < 12.0:
                start_k = b
        end_k = int(seen[-1]) + 1
        result.key_frames["swim_start"] = int(tr.indexes[start_k])
        result.key_frames["finish"] = int(tr.indexes[min(n - 1, end_k - 1)])

        lengths = find_lengths(prep, start_k, end_k)
        tempo, strokes_per = [], []
        all_peaks: list[int] = []
        for i, (s, e, d) in enumerate(lengths):
            count, peaks = count_strokes(prep, s, e)
            seconds = float(tr.times[min(n - 1, e - 1)] - tr.times[s])
            strokes_per.append(int(count))
            tempo.append(round(count / seconds * 60.0, 1) if seconds > 0 and count else None)
            all_peaks.extend(peaks)
            if i > 0:
                result.key_frames[f"length_{i + 1}"] = int(tr.indexes[s])
        total = int(sum(strokes_per))
        swim_s = float(tr.times[min(n - 1, end_k - 1)] - tr.times[start_k])
        result.metrics["stroke_count"] = total
        result.metrics["swim_time_s"] = round(swim_s, 2)
        result.metrics["strokes_by_lap"] = strokes_per
        result.metrics["tempo_by_lap"] = tempo
        if total and swim_s > 0:
            result.metrics["stroke_rate_spm"] = round(total / swim_s * 60.0, 1)
        valid = [t for t in tempo if t]
        if len(valid) >= 2 and valid[0]:
            result.metrics["tempo_change_pct"] = round((valid[-1] - valid[0]) / valid[0] * 100.0, 1)
        elif len(lengths) == 1 and total >= 6:
            # one length: compare the first half with the second half
            s, e, d = lengths[0]
            mid = (s + e) // 2
            c1, _ = count_strokes(prep, s, mid)
            c2, _ = count_strokes(prep, mid, e)
            t1 = float(tr.times[mid] - tr.times[s])
            t2 = float(tr.times[min(n - 1, e - 1)] - tr.times[mid])
            if c1 and t1 > 0 and t2 > 0:
                r1, r2 = c1 / t1, c2 / t2
                result.metrics["tempo_change_pct"] = round((r2 - r1) / r1 * 100.0, 1)
                result.notes.append("swim: tempo change compares the second half of the length with the first half")
        if all_peaks:
            result.key_frames["first_stroke"] = int(tr.indexes[all_peaks[0]])
        result.notes.append("swim: strokes counted from the arm rhythm (each arm for free/back, each cycle for fly/breast)")

        # distance and speed along the lane
        x_m = prep.x_m
        dist = 0.0
        for s, e, d in lengths:
            seg = x_m[s:e]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 2:
                dist += abs(float(seg[-1] - seg[0]))
        if dist > 0 and swim_s > 0:
            result.metrics["avg_speed_mps"] = round(dist / swim_s, 2)
            if total:
                result.metrics["distance_per_stroke_m"] = round(dist / total, 2)
            if prep.cal.confidence == "low":
                result.notes.append("swim: speed and distance per stroke use a guessed scale; calibrate the camera for real numbers")

        # splits from the start signal (or the clip start) at the pool markers
        markers = options.get("markers")
        if markers:
            marks = [float(m) for m in str(markers).split(",")] if isinstance(markers, str) else [float(m) for m in markers]
        else:
            marks = DEFAULT_MARKERS.get(str(options.get("course", "SCY")).upper(), [5.0, 10.0, 15.0])
        signal_start = float(options.get("start_s", 0.0) or 0.0)
        splits: dict[str, float] = {}
        if prep.cal.confidence != "low":
            first_len = lengths[0]
            t = tr.times
            for m in marks:
                if m < pool_m:
                    when = crossing_time(t[first_len[0]:first_len[1]], x_m[first_len[0]:first_len[1]], m, rising=True)
                    if when is not None:
                        splits[f"{m:g}m"] = round(when - signal_start, 2)
            for i, (s, e, d) in enumerate(lengths):
                target = pool_m - 0.5 if d == (prep.direction or 1) else 0.5
                rising = d == (prep.direction or 1)
                when = crossing_time(t[s:e], x_m[s:e], target, rising=rising)
                if when is not None:
                    splits["wall" if i == 0 else f"wall_{i + 1}"] = round(when - signal_start, 2)
            if signal_start <= 0:
                result.notes.append("swim: splits are clip times (no --start given)")
        else:
            result.notes.append("swim: no splits without a calibration")
        result.metrics["splits_s"] = splits or None
        return result
