"""Make a small synthetic clip so a module can be run before real footage exists.

    python tools/make_sample_clip.py                 -> samples/synthetic_sprint.mp4
    python tools/make_sample_clip.py out.mp4 --seconds 12 --fps 30
    python tools/make_sample_clip.py samples/synthetic_100.mp4 --lengths 4 --seconds 40

The clip is a top-down cartoon of one length: a pool, lane lines, a start
signal flash at t = 1.0 s, a "swimmer" blob that leaves the wall, travels
underwater (darker, no splash) for a while, then swims on the surface with a
splash pulse at a fixed stroke rate, and touches the far wall near the end.
With --lengths the swimmer turns at each wall and swims back, with a short
underwater stretch after every push-off, so the whole-event modules have
something to chew on. Nothing about it is realistic.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

WATER = (170, 120, 40)   # BGR
WALL = (200, 200, 200)
LANE = (40, 40, 200)
SWIMMER = (60, 200, 230)
SPLASH = (255, 255, 255)

START_S = 1.0        # start signal
UNDERWATER_S = 3.0   # seconds underwater after the start
UNDERWATER_TURN_S = 1.5  # seconds underwater after each turn
TURN_PAUSE_S = 0.4   # seconds spent at the wall in a turn
STROKE_RATE_SPM = 50


def swimmer_state(race_t: float, seconds: float, lengths: int) -> tuple[float, bool] | None:
    """(progress 0..1 along the lane, underwater?) at race time race_t, or None before the start."""
    if race_t < 0:
        return None
    race_len = seconds - START_S - 0.5
    per_length = (race_len - TURN_PAUSE_S * (lengths - 1)) / lengths
    t = race_t
    for i in range(lengths):
        if t <= per_length or i == lengths - 1:
            frac = min(1.0, t / per_length)
            out = i % 2 == 0
            progress = frac if out else 1.0 - frac
            underwater = t < (UNDERWATER_S if i == 0 else UNDERWATER_TURN_S)
            return progress, underwater
        t -= per_length
        if t <= TURN_PAUSE_S:
            return (1.0 if i % 2 == 0 else 0.0), True
        t -= TURN_PAUSE_S
    return None


_TEXTURE: dict[tuple[int, int], np.ndarray] = {}


def water_texture(width: int, height: int) -> np.ndarray:
    """A fixed speckle so the picture has some texture, like tiles under real water."""
    key = (width, height)
    if key not in _TEXTURE:
        rng = np.random.default_rng(7)
        noise = rng.integers(-18, 19, size=(height, width, 1), dtype=np.int16)
        _TEXTURE[key] = np.clip(np.array(WATER, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return _TEXTURE[key]


def draw_frame(t: float, width: int, height: int, seconds: float, lengths: int = 1) -> np.ndarray:
    frame = water_texture(width, height).copy()
    cv2.rectangle(frame, (0, 0), (30, height), WALL, -1)
    cv2.rectangle(frame, (width - 30, 0), (width, height), WALL, -1)
    for y in range(0, height, height // 6):
        cv2.line(frame, (30, y), (width - 30, y), LANE, 2)

    if START_S <= t < START_S + 0.1:
        cv2.rectangle(frame, (0, 0), (width, height), SPLASH, 12)

    state = swimmer_state(t - START_S, seconds, max(1, lengths))
    if state is not None:
        progress, underwater = state
        x = int(30 + progress * (width - 60))
        y = height // 2
        color = tuple(int(c * 0.6) for c in SWIMMER) if underwater else SWIMMER
        cv2.ellipse(frame, (x, y), (28, 12), 0, 0, 360, color, -1)
        if not underwater:
            phase = ((t - START_S) * STROKE_RATE_SPM / 60.0) % 1.0
            if phase < 0.25:
                cv2.circle(frame, (x + 10, y - 18), 8, SPLASH, -1)
    return frame


def make_clip(path: str, seconds: float = 10.0, fps: float = 30.0, width: int = 640, height: int = 360,
              lengths: int = 1) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        path = os.path.splitext(path)[0] + ".avi"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer on this machine")
    total = int(seconds * fps)
    for index in range(total):
        writer.write(draw_frame(index / fps, width, height, seconds, lengths))
    writer.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=os.path.join("samples", "synthetic_sprint.mp4"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lengths", type=int, default=1, help="number of lengths (2 or more makes a multi-lap event)")
    args = parser.parse_args()
    written = make_clip(args.path, args.seconds, args.fps, lengths=args.lengths)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
