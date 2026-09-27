"""What every module does first: track the swimmer, calibrate, get a speed series.

    prepared = prepare(clip, options, phase="dive")
    prepared.x_m        metres from the start wall, per frame (smoothed)
    prepared.speed      metres per second, per frame
    prepared.track      the raw Track (pixels)
    prepared.cal        the Calibration used
    prepared.view       "underwater" or "above"

The result is cached in options, so when the team app runs all four modules
on one clip the video is only read once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.calibrate import Calibration, background_image, calibrate, water_mask
from core.signal import derivative, fill_gaps, fill_short_gaps, moving_average
from core.track import Track, get_track
from core.video import Clip

MAX_SPEED_MPS = 4.5   # anything faster is a tracking glitch


@dataclass
class Prepared:
    track: Track
    cal: Calibration
    view: str
    times: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    x_m: np.ndarray
    speed: np.ndarray
    direction: int
    surface_y: float | None
    background: np.ndarray
    notes: list[str] = field(default_factory=list)

    def index_at(self, t: float) -> int:
        return self.track.index_at(t)

    def time_at_metre(self, metres: float, after_s: float | None = None) -> float | None:
        """First time the swimmer passes `metres` from the start wall."""
        from core.signal import crossing_time

        pick = np.ones(self.times.size, dtype=bool) if after_s is None else self.times >= after_s
        return crossing_time(self.times[pick], self.x_m[pick], metres, rising=True)

    def mean_speed(self, start_s: float, end_s: float) -> float | None:
        pick = (self.times >= start_s) & (self.times <= end_s) & np.isfinite(self.speed)
        if not pick.any():
            return None
        return float(self.speed[pick].mean())

    def per_second_speed(self) -> list[tuple[float, float]]:
        from core.signal import per_second

        return per_second(self.times, self.speed)


def detect_view(background: np.ndarray) -> str:
    """'underwater' when nearly the whole frame is water, else 'above'."""
    mask = water_mask(background)
    frac = float(mask.mean())
    return "underwater" if frac > 0.62 else "above"


def prepare(clip: Clip, options: dict[str, Any], phase: str = "lap", start_s: float = 0.0, end_s: float | None = None,
            full_frame: bool = False) -> Prepared:
    """Track, calibrate and derive the speed series (cached per clip and window).

    full_frame=True tracks the whole picture instead of one lane band; the dive
    module uses it because the swimmer drops from the block into the water.
    """
    cache = options.setdefault("_prepared", {})
    key = (id(clip), phase, round(start_s, 3), None if end_s is None else round(end_s, 3), full_frame,
           str(options.get("swim_direction", "auto")))
    if key in cache:
        return cache[key]

    track_options = options
    if full_frame:
        track_options = dict(options)
        track_options["full_frame"] = True
        track_options["_tracks"] = options.setdefault("_tracks_full", {})
    track = get_track(clip, track_options, start_s, end_s)
    bg_cache = options.setdefault("_background", {})
    if id(clip) not in bg_cache:
        bg_cache[id(clip)] = background_image(clip)
    background = bg_cache[id(clip)]
    view = str(options.get("view", "auto")).lower()
    if view not in ("underwater", "above"):
        view = detect_view(background)

    x_px = track.smooth_x(0.2)
    y_px = track.smooth_y(0.2)
    known_px = fill_short_gaps(np.where(track.found, 1.0, np.nan), int(round(0.5 * track.fps)))
    x_px = np.where(np.isfinite(known_px), x_px, np.nan)
    y_px = np.where(np.isfinite(known_px), y_px, np.nan)
    forced = str(options.get("swim_direction", "auto")).lower()
    direction = {"right": 1, "left": -1}.get(forced, 0) or _direction_from_track(track, x_px)
    if direction != 0 and "swim_direction" not in options:
        options = dict(options)
        options["swim_direction"] = "right" if direction > 0 else "left"
    cal = calibrate(clip, options, phase=phase, background=background, lane_y=track.lane_y)
    if direction == 0:
        direction = cal.direction

    x_m = np.asarray(cal.x_to_m(x_px), dtype=float)
    # gaps longer than half a second stay unknown instead of being bridged
    x_m = fill_short_gaps(np.where(track.found | np.isfinite(x_px), x_m, np.nan), int(round(0.5 * track.fps)))
    known = fill_short_gaps(np.where(track.found, 1.0, np.nan), int(round(0.5 * track.fps)))
    x_m = np.where(np.isfinite(known), x_m, np.nan)
    speed = derivative(moving_average(fill_gaps(x_m), int(round(track.fps * 0.8))), track.times)
    speed = np.where(np.isfinite(x_m), speed, np.nan)
    speed = np.clip(speed, -MAX_SPEED_MPS, MAX_SPEED_MPS)

    prepared = Prepared(
        track=track, cal=cal, view=view, times=track.times, x_px=x_px, y_px=y_px, x_m=x_m, speed=speed,
        direction=direction, surface_y=cal.surface_y, background=background,
    )
    prepared.notes.extend(track.notes)
    prepared.notes.extend(cal.notes)
    prepared.notes.append(f"camera view: {view}; calibration: {cal.method} ({cal.confidence} confidence)")
    cache[key] = prepared
    return prepared


def _direction_from_track(track: Track, x_px: np.ndarray) -> int:
    ok = track.found & np.isfinite(x_px)
    if ok.sum() < 5:
        return 0
    xs = x_px[ok]
    n = xs.size
    # compare the first quarter with the second quarter, so a clip that
    # includes a turn (swimmer comes back) still reports the outbound direction
    head = np.median(xs[: max(1, n // 4)])
    tail = np.median(xs[max(1, n // 4) : max(2, n // 2)])
    if abs(tail - head) < 0.05 * (np.nanmax(xs) - np.nanmin(xs) + 1e-6):
        return 0
    return 1 if tail > head else -1
