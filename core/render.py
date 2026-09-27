"""Pictures and files made from results: speed chart, PNG report card, PDF, overlay video.

    chart_png = speed_chart_png(profile, phases)
    card_png = report_card_png(results, clip, profile, title)
    pdf_bytes = report_pdf(results, clip, profile, title)
    path = overlay_video(clip, profile, phases, out_path)

`profile` is a SpeedProfile (times, speed, position) pulled out of the shared
analysis cache with speed_profile(options). matplotlib is used for the chart
and the PDF; when it is missing those two are skipped with a note, the PNG
card and the overlay video only need OpenCV.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.report import LABELS, advice_text, all_results, base_module, format_value, is_detail
from core.types import Result
from core.video import Clip

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

PHASE_COLORS = {"dive": "#e07b39", "underwater": "#3a86c8", "swim": "#3fa66b", "turn": "#a45cc4",
                "underwater_after_turn": "#3a86c8", "event": "#c7842b"}
INK = (40, 40, 40)
PAPER = (250, 250, 248)


def phase_color(name: str) -> str:
    return PHASE_COLORS.get(name) or PHASE_COLORS.get(base_module(name), "#666666")


def wrap_text(text: str, width: int) -> list[str]:
    """Break text into lines of at most `width` characters at spaces."""
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


@dataclass
class SpeedProfile:
    times: np.ndarray
    speed: np.ndarray          # m/s, NaN where unknown
    x_m: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    left: np.ndarray
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    found: np.ndarray
    first_index: int
    fps: float
    calibration: str = ""
    notes: list[str] = field(default_factory=list)

    def per_second(self) -> list[tuple[float, float]]:
        from core.signal import per_second

        return per_second(self.times, np.abs(self.speed))


def speed_profile(options: dict[str, Any]) -> SpeedProfile | None:
    """The whole-clip speed series from the analysis cache (after a module ran)."""
    cache = options.get("_prepared") or {}
    best = None
    for key, prep in cache.items():
        # prefer the lane-band track over the full-frame one, and the longest window
        full_frame = bool(key[4]) if len(key) > 4 else False
        score = (0 if full_frame else 1, prep.times.size)
        if best is None or score > best[0]:
            best = (score, prep)
    if best is None:
        return None
    prep = best[1]
    tr = prep.track
    return SpeedProfile(
        times=prep.times, speed=prep.speed, x_m=prep.x_m, x_px=prep.x_px, y_px=prep.y_px,
        left=tr.left, top=tr.top, right=tr.right, bottom=tr.bottom, found=tr.found,
        first_index=tr.first_index, fps=tr.fps, calibration=f"{prep.cal.method} ({prep.cal.confidence})",
    )


def phase_windows(results: list[Result], clip: Clip) -> list[tuple[str, float, float]]:
    """[(phase, start_s, end_s), ...] from the key frames of the results."""
    def t(result: Result, *labels: str) -> float | None:
        for label in labels:
            if label in result.key_frames:
                return clip.frame_to_time(int(result.key_frames[label]))
        return None

    windows: list[tuple[str, float, float]] = []
    for r in all_results(results):
        base = base_module(r.module)
        if base == "dive":
            a, b = t(r, "movement", "start"), t(r, "entry")
        elif base == "underwater":
            a, b = t(r, "entry", "first_kick"), t(r, "breakout")
        elif base == "swim":
            a, b = t(r, "swim_start"), t(r, "finish")
        elif base == "turn":
            a, b = t(r, "turn_in", "wall"), t(r, "turn_out", "wall_leave")
        else:
            a = b = None
        if a is not None and b is not None and b > a:
            windows.append((base, a, b))
    return windows


# ---------------------------------------------------------------- chart

def speed_chart_png(profile: SpeedProfile, phases: list[tuple[str, float, float]] | None = None, width_in: float = 9.0) -> bytes | None:
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(width_in, 3.2), dpi=110)
    speed = np.abs(profile.speed)
    ax.plot(profile.times, speed, color="#333333", linewidth=1.6, label="speed")
    sec = profile.per_second()
    if sec:
        ax.step([s for s, _ in sec] + [sec[-1][0] + 1], [v for _, v in sec] + [sec[-1][1]], where="post",
                color="#888888", linewidth=1.0, alpha=0.8, label="per second")
    for name, a, b in phases or []:
        ax.axvspan(a, b, color=phase_color(name), alpha=0.18, lw=0)
        if len(phases or []) <= 8:
            ax.text((a + b) / 2, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1, name.replace("_", " "),
                    ha="center", va="top", fontsize=8, color=phase_color(name))
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("m/s")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- PNG card

def _put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, color=INK, thick: int = 1) -> int:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return int(28 * scale + 8)


def _thumb(clip: Clip, index: int, width: int) -> np.ndarray:
    frame = clip.frame_at(max(0, min(clip.frame_count - 1, int(index))))
    h = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def report_card_png(results: list[Result], clip: Clip | None, profile: SpeedProfile | None, title: str = "SSAC SplitSecond",
                    chart_png: bytes | None = None) -> bytes:
    """One tall image: title, metrics per phase, key frames, speed chart."""
    width = 1200
    margin = 40
    col_w = (width - 3 * margin) // 2
    blocks: list[np.ndarray] = []

    head = np.full((110, width, 3), PAPER, dtype=np.uint8)
    _put_text(head, title, margin, 55, 1.3, INK, 2)
    sub = f"{os.path.basename(clip.path) if clip is not None and clip.path else ''}"
    if profile is not None:
        sub += f"   calibration: {profile.calibration}"
    _put_text(head, sub.strip(), margin, 92, 0.55, (110, 110, 110))
    blocks.append(head)

    for r in results:
        if is_detail(r):
            continue
        rows = [(LABELS.get(k, k), format_value(v)) for k, v in r.metrics.items() if v is not None]
        notes = [n for n in r.notes if n.startswith(f"{r.module.split('_')[0]}:")]
        frames = list(r.key_frames.items())[:4]
        text_h = 46 + 30 * max(1, len(rows)) + 22 * len(notes)
        thumb_w = (col_w - 12 * (len(frames) - 1)) // max(1, len(frames)) if frames else 0
        thumb_h = int(round(thumb_w * 9 / 16)) + 24 if frames and clip is not None else 0
        h = max(text_h, thumb_h) + 30
        block = np.full((h, width, 3), PAPER, dtype=np.uint8)
        color = phase_color(r.module)
        bgr = tuple(int(color[i : i + 2], 16) for i in (5, 3, 1))
        cv2.rectangle(block, (margin, 12), (margin + 10, h - 12), bgr, -1)
        y = 40
        _put_text(block, r.module.replace("_", " "), margin + 24, y, 0.85, bgr, 2)
        y += 30
        value_x = margin + 24 + max(300, min(430, 40 + 11 * max((len(label) for label, _ in rows), default=0)))
        for label, value in rows:
            _put_text(block, label, margin + 24, y, 0.55)
            _put_text(block, value[:60], value_x, y, 0.55, INK, 1)
            y += 30
        if not rows:
            _put_text(block, "nothing measured", margin + 24, y, 0.55, (140, 140, 140))
            y += 30
        for note in notes:
            _put_text(block, note[:95], margin + 24, y, 0.42, (120, 120, 120))
            y += 22
        if frames and clip is not None:
            x = 2 * margin + col_w
            for label, index in frames:
                thumb = _thumb(clip, index, thumb_w)
                th, tw = thumb.shape[:2]
                block[16 : 16 + th, x : x + tw] = thumb
                _put_text(block, f"{label} ({clip.frame_to_time(int(index)):.2f} s)", x, 16 + th + 18, 0.42, (90, 90, 90))
                x += tw + 12
        blocks.append(block)

    if chart_png is not None:
        chart = cv2.imdecode(np.frombuffer(chart_png, np.uint8), cv2.IMREAD_COLOR)
        if chart is not None:
            scale = (width - 2 * margin) / chart.shape[1]
            chart = cv2.resize(chart, (width - 2 * margin, int(chart.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            block = np.full((chart.shape[0] + 40, width, 3), PAPER, dtype=np.uint8)
            block[20 : 20 + chart.shape[0], margin : margin + chart.shape[1]] = chart
            blocks.append(block)

    blocks.append(_advice_block(results, width, margin))
    foot = np.full((40, width, 3), PAPER, dtype=np.uint8)
    blocks.append(foot)
    card = np.vstack(blocks)
    ok, png = cv2.imencode(".png", card)
    return png.tobytes()


def _advice_block(results: list[Result], width: int, margin: int) -> np.ndarray:
    """Summary and improvement plan as a text block for the PNG card."""
    from core.coach import coach

    advice = coach(results)
    chars = int((width - 2 * margin - 24) / 11)
    lines: list[tuple[str, float, tuple[int, int, int], int]] = [("Summary", 0.85, INK, 2)]
    for sentence in advice.summary:
        lines.extend((line, 0.5, INK, 1) for line in wrap_text(sentence, chars))
    lines.append(("", 0.5, INK, 1))
    lines.append(("Improvement plan (tentative)", 0.85, INK, 2))
    for i, item in enumerate(advice.plan, 1):
        lines.extend((line, 0.55, INK, 2) for line in wrap_text(f"{i}. {item['focus']}: {item['why']}", chars))
        lines.extend((line, 0.5, (90, 90, 90), 1) for line in wrap_text(f"Drill: {item['drill']}", chars))
        lines.extend((line, 0.5, (90, 90, 90), 1) for line in wrap_text(f"Target: {item['target']}", chars))
        lines.append(("", 0.4, INK, 1))
    lines.extend((line, 0.42, (120, 120, 120), 1) for line in wrap_text(advice.basis, int(chars * 1.2)))
    h = 30 + sum(int(28 * scale + 8) for _, scale, _, _ in lines)
    block = np.full((h, width, 3), PAPER, dtype=np.uint8)
    y = 36
    for text, scale, color, thick in lines:
        if text:
            _put_text(block, text, margin + 24, y, scale, color, thick)
        y += int(28 * scale + 8)
    return block


# ---------------------------------------------------------------- PDF

def report_pdf(results: list[Result], clip: Clip | None, profile: SpeedProfile | None, title: str = "SSAC SplitSecond",
               phases: list[tuple[str, float, float]] | None = None) -> bytes | None:
    if not HAVE_MPL:
        return None
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.5, 11), dpi=100)
        fig.text(0.06, 0.96, title, fontsize=18, weight="bold")
        sub = os.path.basename(clip.path) if clip is not None and clip.path else ""
        if profile is not None:
            sub += f"    calibration: {profile.calibration}"
        fig.text(0.06, 0.935, sub, fontsize=9, color="#555555")
        y = 0.90

        def new_page(fig):
            pdf.savefig(fig)
            plt.close(fig)
            return plt.figure(figsize=(8.5, 11), dpi=100)

        for r in all_results(results):
            color = phase_color(r.module)
            fig.text(0.06, y, r.module.replace("_", " "), fontsize=12 if not is_detail(r) else 10, weight="bold", color=color)
            y -= 0.022
            rows = [(LABELS.get(k, k), format_value(v)) for k, v in r.metrics.items() if v is not None]
            if not rows:
                fig.text(0.08, y, "nothing measured", fontsize=9, color="#888888")
                y -= 0.018
            for label, value in rows:
                fig.text(0.08, y, label, fontsize=9)
                for line in wrap_text(value, 70)[:3]:
                    fig.text(0.42, y, line, fontsize=9)
                    y -= 0.017
            for note in r.notes:
                if note.startswith(f"{r.module.split('_')[0]}:"):
                    for line in wrap_text(note, 150)[:2]:
                        fig.text(0.08, y, line, fontsize=7, color="#777777")
                        y -= 0.014
            y -= 0.012
            if y < 0.12:
                fig = new_page(fig)
                y = 0.94
        # summary and plan
        if y < 0.5:
            fig = new_page(fig)
            y = 0.94
        for line in advice_text(results).splitlines():
            text = line.strip()
            if not text:
                y -= 0.008
                continue
            if text.startswith("["):
                y -= 0.01
                fig.text(0.06, y, text.strip("[]").capitalize(), fontsize=12, weight="bold")
                y -= 0.024
                continue
            size = 9 if text[:2].strip().rstrip(".").isdigit() else 8.5
            for piece in wrap_text(text, 120):
                fig.text(0.08, y, piece, fontsize=size, color="#333333" if size == 9 else "#555555")
                y -= 0.016
                if y < 0.06:
                    fig = new_page(fig)
                    y = 0.94
        pdf.savefig(fig)
        plt.close(fig)

        if profile is not None:
            fig, ax = plt.subplots(figsize=(8.5, 11), dpi=100)
            fig.subplots_adjust(top=0.62, bottom=0.42, left=0.1, right=0.95)
            ax.plot(profile.times, np.abs(profile.speed), color="#333333", linewidth=1.5)
            for name, a, b in phases or []:
                ax.axvspan(a, b, color=phase_color(name), alpha=0.18, lw=0)
            ax.set_xlabel("time in clip (s)")
            ax.set_ylabel("speed (m/s)")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.set_title("Speed over the clip")
            sec = profile.per_second()
            if sec:
                text = "   ".join(f"{int(s)}s: {v:.2f}" for s, v in sec)
                fig.text(0.1, 0.36, "Speed by second (m/s):", fontsize=9, weight="bold")
                fig.text(0.1, 0.34, text, fontsize=8, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        if clip is not None:
            frames = [(r.module, label, index) for r in all_results(results) for label, index in r.key_frames.items()][:36]
            for start in range(0, len(frames), 6):
                chunk = frames[start : start + 6]
                fig, axes = plt.subplots(3, 2, figsize=(8.5, 11), dpi=100)
                for ax, (module, label, index) in zip(axes.ravel(), chunk):
                    img = clip.frame_at(max(0, min(clip.frame_count - 1, int(index))))[:, :, ::-1]
                    ax.imshow(img)
                    ax.set_title(f"{module}: {label} ({clip.frame_to_time(int(index)):.2f} s)", fontsize=9)
                    ax.axis("off")
                for ax in axes.ravel()[len(chunk):]:
                    ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- overlay video

def overlay_video(clip: Clip, profile: SpeedProfile | None, phases: list[tuple[str, float, float]], out_path: str,
                  progress=None) -> str:
    """A copy of the clip with speed and phase drawn in the top-left corner and a box on the swimmer."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = os.path.join(tempfile.gettempdir(), f"splitsecond_overlay_{os.getpid()}.mp4")
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*fourcc), clip.fps, (clip.width, clip.height))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError("OpenCV could not open a video writer")
    sec = dict(profile.per_second()) if profile is not None else {}
    pad = max(8, clip.width // 120)
    font = max(0.8, clip.width / 1400)
    for index, t, frame in clip.frames():
        phase = next((name for name, a, b in phases if a <= t <= b), "")
        speed = sec.get(float(int(t)))
        line1 = f"{t:5.2f} s"
        line2 = f"{speed:.2f} m/s" if speed is not None and np.isfinite(speed) else "speed: -"
        box_h = int(90 * font)
        cv2.rectangle(frame, (pad, pad), (pad + int(330 * font), pad + box_h + (int(38 * font) if phase else 0)), (20, 20, 20), -1)
        cv2.putText(frame, line1, (pad + 12, pad + int(34 * font)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (pad + 12, pad + int(72 * font)), cv2.FONT_HERSHEY_SIMPLEX, font, (120, 230, 120), 2, cv2.LINE_AA)
        if phase:
            color = phase_color(phase)
            bgr = tuple(int(color[i : i + 2], 16) for i in (5, 3, 1))
            cv2.putText(frame, phase.replace("_", " "), (pad + 12, pad + int(108 * font)), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, bgr, 2, cv2.LINE_AA)
        if profile is not None:
            k = index - profile.first_index
            if 0 <= k < profile.found.size and profile.found[k]:
                x0, y0, x1, y1 = int(profile.left[k]), int(profile.top[k]), int(profile.right[k]), int(profile.bottom[k])
                cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 220, 255), 2)
        writer.write(frame)
        if progress is not None and index % 30 == 0:
            progress(index / max(1, clip.frame_count))
    writer.release()
    # browsers want H.264; use ffmpeg when it is around, otherwise keep what OpenCV wrote
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        done = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", out_path])
        if done.returncode == 0 and os.path.exists(out_path):
            os.remove(tmp_path)
            return out_path
    shutil.move(tmp_path, out_path)
    return out_path


COMPARE_COLORS = ["#333333", "#e07b39", "#3a86c8", "#3fa66b"]


def compare_chart_png(profiles: list[tuple[str, SpeedProfile]], align_s: list[float] | None = None) -> bytes | None:
    """Speed curves of several clips on one chart. align_s shifts each clip so its
    chosen moment (the entry, the breakout) sits at zero."""
    if not HAVE_MPL or not profiles:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 3.4), dpi=110)
    for i, (label, prof) in enumerate(profiles):
        shift = align_s[i] if align_s and i < len(align_s) and align_s[i] is not None else 0.0
        ax.plot(prof.times - shift, np.abs(prof.speed), color=COMPARE_COLORS[i % len(COMPARE_COLORS)], linewidth=1.6, label=label)
    ax.set_xlabel("time (s)" + (", aligned" if align_s and any(a for a in align_s) else " in clip"))
    ax.set_ylabel("m/s")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
