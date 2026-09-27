"""Finding the swimmer in every frame.

All four modules need the same thing: where the swimmer is, frame by frame.
This module does it once and hands back a Track. It works from motion, so it
needs no model files and runs on a laptop:

  1. Camera shift between frames is measured with phase correlation, so a
     phone that drifts or pans a little does not turn into swimmer motion.
  2. Foreground = pixels that changed since the last frame AND differ from a
     slowly updated background model. Water sparkle changes every frame but
     is not in the background model for long; the swimmer is both.
  3. The lane band (rows the swimmer is in) is either given by the coach
     (options["swimmer_box"] or options["lane_y"]) or found from where the
     motion energy is concentrated over the whole clip.
  4. In the band, the biggest moving blob near the predicted position is the
     swimmer. Small gaps are bridged; long gaps are left as NaN.

The Track keeps per-frame centroid, bounding box, area and body angle, in
full-resolution pixel coordinates of a fixed reference frame (camera shift
already removed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.signal import fill_gaps, median_filter, moving_average
from core.video import Clip

WORK_WIDTH = 640          # frames are shrunk to this width for detection
MIN_BLOB_FRACTION = 0.00008  # of the work frame area


@dataclass
class Track:
    fps: float
    first_index: int
    times: np.ndarray                     # seconds in the clip
    cx: np.ndarray                        # centroid column, full-res px, camera shift removed
    cy: np.ndarray                        # centroid row
    left: np.ndarray
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    area: np.ndarray                      # px^2 in full-res units
    angle: np.ndarray                     # body axis vs horizontal, degrees, positive = head lower than feet toward travel direction
    found: np.ndarray                     # bool
    white: np.ndarray                     # bright white pixels (foam, splash) inside the blob box, work-scale px
    cam_dx: np.ndarray                    # accumulated camera shift (px) removed from x
    lane_y: tuple[float, float]           # band of rows the swimmer was tracked in
    scale: float
    notes: list[str] = field(default_factory=list)

    @property
    def indexes(self) -> np.ndarray:
        return self.first_index + np.arange(self.times.size)

    def index_at(self, t: float) -> int:
        """Clip frame index closest to time t."""
        k = int(np.argmin(np.abs(self.times - t)))
        return int(self.indexes[k])

    def window(self, start_s: float | None, end_s: float | None) -> np.ndarray:
        pick = np.ones(self.times.size, dtype=bool)
        if start_s is not None:
            pick &= self.times >= start_s
        if end_s is not None:
            pick &= self.times <= end_s
        return pick

    def smooth_x(self, seconds: float = 0.2) -> np.ndarray:
        x = np.where(self.found, self.cx, np.nan)
        x = median_filter(x, int(round(self.fps * 0.1)))
        return fill_gaps(moving_average(x, int(round(self.fps * seconds))))

    def smooth_y(self, seconds: float = 0.2) -> np.ndarray:
        y = np.where(self.found, self.cy, np.nan)
        y = median_filter(y, int(round(self.fps * 0.1)))
        return fill_gaps(moving_average(y, int(round(self.fps * seconds))))

    def slice(self, start_s: float | None, end_s: float | None) -> "Track":
        """The part of the track inside [start_s, end_s], as its own Track."""
        pick = self.window(start_s, end_s)
        if pick.all():
            return self
        idx = np.where(pick)[0]
        if idx.size == 0:
            idx = np.array([0])
        a, b = int(idx[0]), int(idx[-1]) + 1
        return Track(
            fps=self.fps, first_index=self.first_index + a, times=self.times[a:b], cx=self.cx[a:b], cy=self.cy[a:b],
            left=self.left[a:b], top=self.top[a:b], right=self.right[a:b], bottom=self.bottom[a:b], area=self.area[a:b],
            angle=self.angle[a:b], found=self.found[a:b], white=self.white[a:b], cam_dx=self.cam_dx[a:b],
            lane_y=self.lane_y, scale=self.scale, notes=list(self.notes),
        )

    def clean(self, frame_width: float, frame_area: float) -> None:
        """Drop detections that are too small to be a swimmer or that teleport.

        A blob that jumps more than 3% of the frame width per frame away from
        the last accepted position is noise (sparkle, a person on deck), unless
        the jumps keep coming for half a second, in which case the earlier
        anchor was the noise and the track re-anchors on the new position.
        """
        min_area = 0.0002 * frame_area
        if self.found.any():
            # relative gate: the swimmer is the big thing; blobs a quarter of
            # that size are sparkle, reflections or someone far away
            min_area = max(min_area, 0.35 * float(np.percentile(self.area[self.found], 90)))
        ok = self.found & (self.area >= min_area)
        max_step = 0.06 * frame_width
        anchor_x = None
        anchor_k = None
        rejected = 0
        keep = np.zeros_like(ok)
        for k in range(ok.size):
            if not ok[k]:
                continue
            if anchor_x is None:
                keep[k] = True
                anchor_x, anchor_k, rejected = self.cx[k], k, 0
                continue
            gap = max(1, k - anchor_k)
            if abs(self.cx[k] - anchor_x) <= max_step * gap:
                keep[k] = True
                anchor_x, anchor_k, rejected = self.cx[k], k, 0
            else:
                rejected += 1
                if rejected >= int(round(0.5 * self.fps)):
                    keep[k] = True
                    anchor_x, anchor_k, rejected = self.cx[k], k, 0
        self.found = keep

    def coverage(self) -> float:
        return float(self.found.mean()) if self.found.size else 0.0


def _shrink(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _band_from_options(options: dict[str, Any], height: int) -> tuple[float, float] | None:
    box = options.get("swimmer_box")
    if box and len(box) == 4:
        x, y, w, h = [float(v) for v in box]
        return (max(0.0, y - 0.5 * h), min(float(height), y + 1.5 * h))
    band = options.get("lane_y")
    if band:
        if isinstance(band, str):
            a, b = band.split(":")
            return (float(a), float(b))
        return (float(band[0]), float(band[1]))
    return None


_WINDOWS: dict[tuple[int, int], np.ndarray] = {}


def _camera_shift(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Horizontal shift of the whole frame between two grayscale images, in px."""
    a = np.float32(prev_gray)
    b = np.float32(gray)
    size = (a.shape[1], a.shape[0])
    win = _WINDOWS.get(size)
    if win is None:
        win = _WINDOWS[size] = cv2.createHanningWindow(size, cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    if response < 0.05:
        return 0.0
    return float(dx)


def track_swimmer(clip: Clip, options: dict[str, Any], start_s: float = 0.0, end_s: float | None = None,
                  progress=None) -> Track:
    """Run the tracker over [start_s, end_s]. Never raises for an ordinary clip."""
    scale = min(1.0, WORK_WIDTH / float(clip.width))
    height = int(round(clip.height * scale))
    width = int(round(clip.width * scale))
    band = _band_from_options(options, clip.height)
    if options.get("full_frame"):
        band = (0.0, float(clip.height))
    band_small = (band[0] * scale, band[1] * scale) if band else None

    first = clip.time_to_frame(start_s)
    last = clip.frame_count - 1 if end_s is None else clip.time_to_frame(end_s)
    count = max(1, last - first + 1)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)

    # pass 1 (only when no band was given): which rows hold a thing that moves
    # steadily across the picture? Every moving blob of every frame is kept,
    # then candidate bands are scored by how smooth and how far the biggest
    # blob inside them travels. Water sparkle jumps around; the swimmer glides.
    if band_small is None:
        detector = _Foreground(clip.fps)
        blobs: list[np.ndarray] = []
        for index, _, frame in clip.frames(start_s=start_s, end_s=end_s):
            m = detector.apply(_shrink(frame, scale))
            if m is None:
                blobs.append(np.zeros((0, 3)))
                continue
            n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
            keep = [(cents[j][0], cents[j][1], float(stats[j, cv2.CC_STAT_AREA])) for j in range(1, n)
                    if stats[j, cv2.CC_STAT_AREA] >= MIN_BLOB_FRACTION * width * height]
            blobs.append(np.array(keep, dtype=float).reshape(-1, 3))
        band_small = _pick_band_from_blobs(blobs, width, height)

    y0, y1 = int(max(0, band_small[0])), int(min(height, band_small[1]))
    min_area = MIN_BLOB_FRACTION * width * height
    detector = _Foreground(clip.fps)

    times = np.zeros(count)
    cx = np.full(count, np.nan)
    cy = np.full(count, np.nan)
    left = np.full(count, np.nan)
    top = np.full(count, np.nan)
    right = np.full(count, np.nan)
    bottom = np.full(count, np.nan)
    area = np.zeros(count)
    angle = np.full(count, np.nan)
    found = np.zeros(count, dtype=bool)
    panning = np.zeros(count, dtype=bool)
    white = np.zeros(count)
    cam_dx = np.zeros(count)

    prev_gray = None
    shift_total = 0.0
    last_pos: tuple[float, float] | None = None
    last_seen = -1
    k = 0
    for index, t, frame in clip.frames(start_s=start_s, end_s=end_s):
        small = _shrink(frame, scale)
        blur = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = detector.apply(small)
        times[k] = t
        if prev_gray is not None and mask is not None:
            dx = _camera_shift(prev_gray, blur)
            if abs(dx) > 0.75:
                shift_total += dx
            cam_dx[k] = shift_total / scale
            panning[k] = abs(dx) > 0.004 * width
            mask[:y0] = 0
            mask[y1:] = 0
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            choice = _choose_blob(stats, cents, n, min_area, last_pos, last_seen, k, clip.fps)
            if choice is not None:
                sx, sy, sw, sh, sa = stats[choice, :5]
                ccx, ccy = cents[choice]
                found[k] = True
                cx[k] = ccx / scale - cam_dx[k]
                cy[k] = ccy / scale
                left[k] = sx / scale - cam_dx[k]
                top[k] = sy / scale
                right[k] = (sx + sw) / scale - cam_dx[k]
                bottom[k] = (sy + sh) / scale
                area[k] = sa / (scale * scale)
                angle[k] = _blob_angle(labels == choice)
                white[k] = _white_count(small, detector.background(), int(sx), int(sy), int(sw), int(sh))
                last_pos = (float(ccx), float(ccy))
                last_seen = k
        prev_gray = blur
        k += 1
        if progress is not None and k % 30 == 0:
            progress(k / count)

    track = Track(
        fps=clip.fps, first_index=first, times=times, cx=cx, cy=cy, left=left, top=top, right=right, bottom=bottom,
        area=area, angle=angle, found=found, white=white, cam_dx=cam_dx,
        lane_y=(band_small[0] / scale, band_small[1] / scale), scale=scale,
    )
    # while the camera pans (and for a moment after, while the background
    # model catches up) every pixel moves, so nothing found then is trusted
    if panning.any():
        settle = int(round(0.7 * clip.fps))
        blocked = panning.copy()
        for k in np.where(panning)[0]:
            blocked[k : k + settle] = True
        track.found &= ~blocked
        track.notes.append(f"camera panned for {panning.sum() / clip.fps:.1f} s; those frames were skipped")
    track.clean(clip.width, clip.width * clip.height)
    if track.coverage() < 0.3:
        track.notes.append(f"swimmer found in only {100 * track.coverage():.0f}% of frames; results may be rough")
    if np.abs(cam_dx).max() > 0.02 * clip.width:
        track.notes.append(f"camera moved about {np.abs(cam_dx).max():.0f} px during the clip; motion was compensated")
    return track


class _Foreground:
    """Moving pixels that are not water sparkle.

    A MOG2 background model flags anything that differs from the recent
    background. Water sparkle flickers on and off in the same place all the
    time, so a per-pixel "how often is this foreground" average separates it
    from the swimmer, who passes through a pixel once.
    """

    def __init__(self, fps: float):
        self.sub = cv2.createBackgroundSubtractorMOG2(history=int(4 * fps), varThreshold=28, detectShadows=False)
        self.rate: np.ndarray | None = None
        self.alpha = 1.0 / max(10.0, 1.5 * fps)
        self.count = 0
        self.kernel3 = np.ones((3, 3), np.uint8)
        self.kernel7 = np.ones((7, 7), np.uint8)

    def background(self) -> np.ndarray | None:
        try:
            return self.sub.getBackgroundImage()
        except cv2.error:
            return None

    def apply(self, small: np.ndarray) -> np.ndarray | None:
        fg = self.sub.apply(small)
        self.count += 1
        if self.count < 3:
            # the first MOG2 frames are all foreground while the model fills
            self.rate = np.zeros(fg.shape, dtype=np.float32)
            return None
        on = (fg > 0).astype(np.float32)
        self.rate += self.alpha * (on - self.rate)
        steady = self.rate > 0.3
        fg[steady] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel3)
        fg = cv2.dilate(fg, self.kernel7)
        return fg


def _pick_band_from_blobs(blobs: list[np.ndarray], width: int, height: int) -> tuple[float, float]:
    """Score sliding row bands by the steady travel of the biggest blob inside them."""
    band_h = max(8, int(round(0.10 * height)))
    step = max(2, band_h // 4)
    best: tuple[float, float, float] | None = None
    for y0 in range(0, max(1, height - band_h + 1), step):
        y1 = y0 + band_h
        xs = np.full(len(blobs), np.nan)
        for k, b in enumerate(blobs):
            if b.size == 0:
                continue
            inside = b[(b[:, 1] >= y0) & (b[:, 1] < y1)]
            if inside.size:
                xs[k] = inside[np.argmax(inside[:, 2]), 0]
        ok = np.isfinite(xs)
        if ok.sum() < 5:
            continue
        # smooth pairs: consecutive detections that moved less than 4% of the width
        both = ok[1:] & ok[:-1]
        dx = np.abs(np.diff(xs))
        smooth = float(np.sum(both & (dx < 0.04 * width)))
        travel = float(np.nanpercentile(xs, 95) - np.nanpercentile(xs, 5)) / width
        score = smooth * (0.2 + travel)
        if best is None or score > best[0]:
            best = (score, float(y0), float(y1))
    if best is None:
        return (0.0, float(height))
    pad = band_h // 2
    return (max(0.0, best[1] - pad), min(float(height), best[2] + pad))


def _choose_blob(stats, cents, n: int, min_area: float, last_pos, last_seen: int, k: int, fps: float):
    if n <= 1:
        return None
    best = None
    best_score = -1.0
    for j in range(1, n):
        a = float(stats[j, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        score = a
        if last_pos is not None:
            gap = max(1, k - last_seen)
            dist = float(np.hypot(cents[j][0] - last_pos[0], cents[j][1] - last_pos[1]))
            allowed = 40.0 * gap + 40.0     # px per frame at work scale, generous for a sprinter
            if dist > allowed and gap < fps:
                score *= 0.15
            else:
                score *= 1.0 / (1.0 + dist / 80.0)
        if score > best_score:
            best_score = score
            best = j
    return best


def _white_count(small: np.ndarray, background: np.ndarray | None, x: int, y: int, w: int, h: int) -> float:
    """How many bright, colourless pixels (foam) are inside a box and were not
    white in the background (so white tiles and doors do not count)."""
    patch = small[max(0, y) : y + h, max(0, x) : x + w]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 185)
    if background is not None and background.shape == small.shape:
        bg = cv2.cvtColor(background[max(0, y) : y + h, max(0, x) : x + w], cv2.COLOR_BGR2HSV)
        white &= ~((bg[:, :, 1] < 60) & (bg[:, :, 2] > 165))
    return float(white.sum())


def _blob_angle(mask: np.ndarray) -> float:
    """Orientation of the blob's long axis in degrees from horizontal (0..90)."""
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return float("nan")
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return float(ang)


def get_track(clip: Clip, options: dict[str, Any], start_s: float = 0.0, end_s: float | None = None) -> Track:
    """Tracker with a cache in options, so several modules share one pass."""
    cache = options.setdefault("_tracks", {})
    key = id(clip)
    if key not in cache:
        cache[key] = track_swimmer(clip, options, 0.0, None, progress=options.get("_progress"))
    return cache[key].slice(start_s if start_s > 0 else None, end_s)
