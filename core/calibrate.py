"""Turning pixels into metres.

Every distance and speed number depends on this, so it is shared by all
modules. A Calibration maps a horizontal pixel position in the frame to a
distance in metres from the start wall, along the swimmer's lane.

Where the numbers come from, in order of preference:

  1. Reference points the coach marked once for this camera position
     (options["calib_points"] = "x1:m1,x2:m2[,x3:m3...]" or a saved profile
     name in options["calib_profile"]). Two or more points; more points
     handle perspective better. The pool's foot ruler on the gutter, the
     backstroke flags (5 yd or 5 m from the wall) and the 15 m mark all
     make good reference points.
  2. Both end walls visible in the frame: the water in the swimmer's lane
     runs from wall to wall and that is the pool length.
  3. options["frame_span_m"]: how many metres the frame is wide.
  4. A guess, flagged low confidence, so the module still runs.

Profiles live in calibration/<name>.json next to run.py and are written by
the web page, so a camera position is calibrated once and reused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.paths import ROOT
from core.video import Clip

PROFILES_DIR = ROOT / "calibration"

DEFAULT_SPAN_M = {"dive": 8.0, "underwater": 15.0, "swim": 25.0, "turn": 10.0, "lap": 25.0}


@dataclass
class Calibration:
    points: list[tuple[float, float]]          # (x_px, metres from start wall), sorted by x
    direction: int = 1                          # +1 swimmer travels toward larger x
    lane_y: tuple[float, float] | None = None   # rows (top, bottom) of the swimmer's lane
    surface_y: float | None = None              # row of the water surface at the lane, if known
    wall_x: float | None = None                 # column of the start wall, if known
    method: str = "default"
    confidence: str = "low"                     # high / medium / low
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = sorted((float(x), float(m)) for x, m in self.points)
        if len(self.points) >= 2:
            first, last = self.points[0], self.points[-1]
            self.direction = 1 if last[1] >= first[1] else -1
        if self.wall_x is None and self.points:
            # the point closest to 0 m is the wall if it is at 0
            for x, m in self.points:
                if abs(m) < 1e-6:
                    self.wall_x = x

    def x_to_m(self, x_px: float | np.ndarray) -> float | np.ndarray:
        xs = np.array([p[0] for p in self.points])
        ms = np.array([p[1] for p in self.points])
        if xs.size == 0:
            return np.nan * np.asarray(x_px, dtype=float)
        if xs.size == 1:
            return ms[0] + 0.0 * np.asarray(x_px, dtype=float)
        x = np.asarray(x_px, dtype=float)
        inside = np.interp(x, xs, ms)
        # linear extrapolation beyond the outer points using the outer segments
        left_slope = (ms[1] - ms[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
        right_slope = (ms[-1] - ms[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
        out = np.where(x < xs[0], ms[0] + (x - xs[0]) * left_slope, inside)
        out = np.where(x > xs[-1], ms[-1] + (x - xs[-1]) * right_slope, out)
        return float(out) if np.ndim(x_px) == 0 else out

    def m_per_px(self, x_px: float | None = None) -> float:
        """Metres per pixel near x_px (or the average over the marked span)."""
        if len(self.points) < 2:
            return float("nan")
        if x_px is None:
            (x0, m0), (x1, m1) = self.points[0], self.points[-1]
            return abs((m1 - m0) / (x1 - x0)) if x1 != x0 else float("nan")
        a = self.x_to_m(x_px - 5.0)
        b = self.x_to_m(x_px + 5.0)
        return abs(float(b) - float(a)) / 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [[x, m] for x, m in self.points],
            "direction": self.direction,
            "lane_y": list(self.lane_y) if self.lane_y else None,
            "surface_y": self.surface_y,
            "wall_x": self.wall_x,
            "method": self.method,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        return cls(
            points=[(float(x), float(m)) for x, m in data.get("points", [])],
            direction=int(data.get("direction", 1)),
            lane_y=tuple(data["lane_y"]) if data.get("lane_y") else None,
            surface_y=data.get("surface_y"),
            wall_x=data.get("wall_x"),
            method=data.get("method", "profile"),
            confidence=data.get("confidence", "high"),
        )


# ---------------------------------------------------------------- profiles

def profile_path(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return str(PROFILES_DIR / f"{safe}.json")


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PROFILES_DIR) if f.endswith(".json"))


def save_profile(name: str, calibration: Calibration) -> str:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2)
    return path


def load_profile(name: str) -> Calibration | None:
    path = profile_path(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        cal = Calibration.from_dict(json.load(handle))
    cal.method = f"profile:{name}"
    return cal


# ---------------------------------------------------------------- background and water

def background_image(clip: Clip, samples: int = 24, start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    """Median of evenly spaced frames: the pool without the swimmer."""
    first = clip.time_to_frame(start_s)
    last = clip.frame_count - 1 if end_s is None else clip.time_to_frame(end_s)
    count = max(1, min(samples, last - first + 1))
    indexes = np.linspace(first, last, count).astype(int)
    frames = [clip.frame_at(int(i)) for i in indexes]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def water_mask(image: np.ndarray) -> np.ndarray:
    """1 where the pixel looks like pool water (cyan to blue), else 0."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h >= 70) & (h <= 110) & (s >= 45) & (v >= 50)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


def longest_run(row: np.ndarray) -> tuple[int, int] | None:
    """(start, end) of the longest run of True values in a 1-D bool array."""
    best: tuple[int, int] | None = None
    start = None
    for i, on in enumerate(list(row) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def water_extent(mask: np.ndarray, lane_y: tuple[float, float]) -> tuple[int, int] | None:
    """Leftmost and rightmost water columns of the lane, or None.

    Each row of the band is checked on its own and the rows whose water run
    ends inside the picture (both walls visible) vote with their median.
    """
    y0, y1 = int(max(0, lane_y[0])), int(min(mask.shape[0], lane_y[1]))
    if y1 <= y0:
        return None
    width = mask.shape[1]
    lefts, rights = [], []
    for y in range(y0, y1, 2):
        run = longest_run(mask[y] > 0)
        if run is None:
            continue
        left, right = run
        if left <= 2 or right >= width - 3 or right - left < 0.5 * width:
            continue
        lefts.append(left)
        rights.append(right)
    if len(lefts) < 3:
        return None
    return int(np.median(lefts)), int(np.median(rights))


def surface_row(mask: np.ndarray, x_px: float, lane_y: tuple[float, float] | None = None) -> float | None:
    """Row where water starts (top edge of the water) in the columns around x_px."""
    x0, x1 = int(max(0, x_px - 40)), int(min(mask.shape[1], x_px + 40))
    if x1 <= x0:
        return None
    col = mask[:, x0:x1].mean(axis=1) > 0.5
    top = 0 if lane_y is None else int(max(0, lane_y[0] - 200))
    for y in range(top, mask.shape[0]):
        if col[y]:
            return float(y)
    return None


# ---------------------------------------------------------------- building a calibration

def parse_points(text: Any) -> list[tuple[float, float]]:
    """'x1:m1,x2:m2' or [[x, m], ...] into a list of (x, m)."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [(float(x), float(m)) for x, m in text]
    points = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        x, m = item.split(":")
        points.append((float(x), float(m)))
    return points


def calibrate(clip: Clip, options: dict[str, Any], phase: str = "lap", background: np.ndarray | None = None,
              lane_y: tuple[float, float] | None = None) -> Calibration:
    """Pick the best available calibration for this clip. Never raises."""
    pool_m = float(options.get("pool_length_m", 22.86))
    lane_y = lane_y or _lane_from_options(options, clip)
    notes: list[str] = []

    # 1. explicit points or a saved profile
    points = parse_points(options.get("calib_points"))
    if len(points) >= 2:
        cal = Calibration(points, lane_y=lane_y, method="points", confidence="high")
        _fill_surface(cal, clip, background, lane_y)
        return cal
    name = options.get("calib_profile")
    if name:
        cal = load_profile(str(name))
        if cal is not None:
            if lane_y and not cal.lane_y:
                cal.lane_y = lane_y
            _fill_surface(cal, clip, background, cal.lane_y)
            return cal
        notes.append(f"calibration profile '{name}' not found")

    bg = background if background is not None else background_image(clip)
    mask = water_mask(bg)

    # 2. both walls visible
    if lane_y is not None:
        run = water_extent(mask, lane_y)
        if run is not None:
            left, right = run
            if True:
                direction = _direction_from_options(options)
                if direction >= 0:
                    points = [(float(left), 0.0), (float(right), pool_m)]
                else:
                    points = [(float(right), 0.0), (float(left), pool_m)]
                cal = Calibration(points, lane_y=lane_y, method="walls", confidence="medium", notes=notes)
                cal.notes.append("calibrated from the two end walls; check that both walls really are in frame")
                _fill_surface(cal, clip, bg, lane_y)
                return cal

    # 3. frame span given
    span = options.get("frame_span_m")
    if span:
        span = float(span)
        direction = _direction_from_options(options)
        direction = 1 if direction == 0 else direction
        origin = float(options.get("wall_x", 0.0 if direction > 0 else clip.width))
        if direction > 0:
            points = [(origin, 0.0), (origin + clip.width, span)]
        else:
            points = [(origin, 0.0), (origin - clip.width, span)]
        cal = Calibration(points, lane_y=lane_y, method="frame_span", confidence="medium", notes=notes)
        _fill_surface(cal, clip, bg, lane_y)
        return cal

    # 4. guess
    span = DEFAULT_SPAN_M.get(phase, 25.0)
    direction = _direction_from_options(options)
    direction = 1 if direction == 0 else direction
    if direction > 0:
        points = [(0.0, 0.0), (float(clip.width), span)]
    else:
        points = [(float(clip.width), 0.0), (0.0, span)]
    cal = Calibration(points, lane_y=lane_y, method="default", confidence="low", notes=notes)
    cal.notes.append(
        f"no calibration: assumed the frame spans {span:.0f} m. Distances and speeds are rough. "
        "Mark two reference points (Calibrate on the web page, or -o calib_points=x1:m1,x2:m2)."
    )
    _fill_surface(cal, clip, bg, lane_y)
    return cal


def _lane_from_options(options: dict[str, Any], clip: Clip) -> tuple[float, float] | None:
    box = options.get("swimmer_box")
    if box and len(box) == 4:
        x, y, w, h = box
        return (float(y) - h, float(y) + 2 * h)
    band = options.get("lane_y")
    if band:
        if isinstance(band, str):
            a, b = band.split(":")
            return (float(a), float(b))
        return (float(band[0]), float(band[1]))
    return None


def _direction_from_options(options: dict[str, Any]) -> int:
    text = str(options.get("swim_direction", "auto")).lower()
    if text in ("right", "r", "+1", "1"):
        return 1
    if text in ("left", "l", "-1"):
        return -1
    return 0


def _fill_surface(cal: Calibration, clip: Clip, background: np.ndarray | None, lane_y: tuple[float, float] | None) -> None:
    if cal.surface_y is not None or lane_y is None:
        return
    try:
        bg = background if background is not None else background_image(clip, samples=8)
        mask = water_mask(bg)
        x = cal.wall_x if cal.wall_x is not None else clip.width / 2
        x = min(max(40.0, float(x)), clip.width - 40.0)
        cal.surface_y = surface_row(mask, x, lane_y)
    except Exception:
        cal.surface_y = None
