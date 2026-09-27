"""Turning Results into something a person can read.

Plain text for the terminal, JSON for saving, plain-language metric labels
for the web page and the reports, and the summary / improvement plan text
(from core/coach.py) that goes at the end of every report.
"""

from __future__ import annotations

import json
from typing import Any

from core.types import Result

# Plain-language names for the metric keys, used on the web page and in the reports.
LABELS: dict[str, str] = {
    "reaction_time_s": "Reaction time (s)",
    "flight_time_s": "Flight time (s)",
    "entry_distance_m": "Entry distance from wall (m)",
    "entry_angle_deg": "Entry angle, body (deg)",
    "entry_path_angle_deg": "Entry angle, flight path (deg)",
    "entry_depth_m": "Entry depth (m)",
    "movement_time_s": "First movement (s in clip)",
    "entry_time_s": "Entry (s in clip)",
    "underwater_distance_m": "Underwater distance (m)",
    "underwater_time_s": "Underwater time (s)",
    "kick_count": "Kicks",
    "breakout_time_s": "Breakout time (s)",
    "breakout_speed_mps": "Breakout speed (m/s)",
    "breakout_slowdown_pct": "Slowdown before breakout (%)",
    "peak_underwater_speed_mps": "Peak underwater speed (m/s)",
    "stroke_count": "Strokes",
    "stroke_rate_spm": "Tempo (strokes/min)",
    "distance_per_stroke_m": "Distance per stroke (m)",
    "avg_speed_mps": "Average speed (m/s)",
    "swim_time_s": "Swim time (s)",
    "splits_s": "Splits (s)",
    "tempo_by_lap": "Tempo by length (strokes/min)",
    "strokes_by_lap": "Strokes by length",
    "tempo_change_pct": "Tempo change (%)",
    "turn_type": "Turn type",
    "turn_in_time_s": "Turn in: zone to wall (s)",
    "turn_out_time_s": "Turn out: wall to zone (s)",
    "turn_time_s": "Turn time (s)",
    "wall_contact_s": "Time on the wall (s)",
    "push_off_speed_mps": "Push-off speed (m/s)",
    "turn_zone_m": "Turn zone each side (m)",
    "lengths": "Lengths",
    "distance_m": "Distance (m)",
    "event_time_s": "Event time (s)",
    "split_by_length_s": "Split by length (s)",
    "cumulative_split_s": "Time at each wall (s)",
    "speed_by_length_mps": "Speed by length (m/s)",
    "speed_change_pct": "Speed change, last length (%)",
    "strokes_by_length": "Strokes by length",
    "tempo_by_length_spm": "Tempo by length (strokes/min)",
    "distance_per_stroke_by_length_m": "Distance per stroke by length (m)",
    "surface_speed_by_length_mps": "Surface speed by length (m/s)",
    "strokes_change_pct": "Stroke count change, last length (%)",
    "underwater_distance_by_length_m": "Underwater distance by length (m)",
    "underwater_time_by_length_s": "Underwater time by length (s)",
    "underwater_change_pct": "Underwater distance change, last length (%)",
    "kicks_by_length": "Kicks by length",
    "breakout_speed_by_length_mps": "Breakout speed by length (m/s)",
    "breakout_slowdown_by_length_pct": "Slowdown before breakout by length (%)",
    "peak_underwater_speed_by_length_mps": "Peak underwater speed by length (m/s)",
    "turn_count": "Turns",
    "turn_time_by_turn_s": "Turn time by turn (s)",
    "turn_time_avg_s": "Average turn time (s)",
    "wall_contact_by_turn_s": "Time on the wall by turn (s)",
    "push_off_speed_by_turn_mps": "Push-off speed by turn (m/s)",
}


def is_detail(result: Result) -> bool:
    """True for a per-length result of a whole-event run (swim_L3, turn_L2 ...)."""
    import re

    return re.search(r"_L\d+$", result.module) is not None


def base_module(name: str) -> str:
    """'swim' for 'swim_L3', 'underwater' for 'underwater_after_turn', 'event' for 'event_above'."""
    return name.split("_L")[0].split("_after")[0] if not name.startswith("event") else "event"


def all_results(results: list[Result]) -> list[Result]:
    """The results plus every detail behind them, in reading order."""
    out: list[Result] = []
    for r in results:
        out.append(r)
        out.extend(r.details)
    return out


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return ", ".join(f"{k} {format_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    return str(value)


def to_text(results: list[Result], with_advice: bool = True) -> str:
    lines: list[str] = []
    for result in all_results(results):
        lines.append(f"[{result.module}]")
        width = max((len(key) for key in result.metrics), default=0)
        for key, value in result.metrics.items():
            lines.append(f"  {key.ljust(width)}  {format_value(value)}")
        if result.key_frames:
            frames = ", ".join(f"{label}={index}" for label, index in result.key_frames.items())
            lines.append(f"  key frames: {frames}")
        for note in result.notes:
            lines.append(f"  note: {note}")
        lines.append("")
    if with_advice:
        lines.extend(advice_text(results).splitlines())
    return "\n".join(lines).rstrip() + "\n"


def advice_text(results: list[Result]) -> str:
    """The summary and the plan as plain text (terminal, PDF)."""
    from core.coach import coach

    advice = coach(results)
    lines = ["[summary]"]
    lines.extend(f"  {line}" for line in advice.summary)
    lines.append("")
    lines.append("[improvement plan, tentative]")
    for i, item in enumerate(advice.plan, 1):
        lines.append(f"  {i}. {item['focus']}: {item['why']}")
        lines.append(f"     drill: {item['drill']}")
        lines.append(f"     target: {item['target']}")
    lines.append(f"  {advice.basis}")
    return "\n".join(lines) + "\n"


def to_json(results: list[Result], clip_path: str = "") -> str:
    from core.coach import coach

    payload = {"clip": clip_path, "results": [result.to_dict() for result in results], "advice": coach(results).to_dict()}
    return json.dumps(payload, indent=2)


# What to tell the coach when the chosen phase is not in the video at all.
PHASE_MESSAGES: dict[str, str] = {
    "dive": "Please upload a video that includes the start: the swimmer on the block and the dive into the water.",
    "underwater": "Please upload a video that shows the swimmer underwater through to the breakout.",
    "swim": "Please upload a video that shows the swimmer swimming on the surface for at least a few strokes.",
    "turn": "Please upload a video that includes a turn: the swimmer reaching the wall and pushing off again.",
    "lap": "Please upload a video that shows a whole lap: the swimmer moving down the lane from one wall to the other.",
    "event_above": "Please upload a video that shows the whole swim from an above-water camera: the swimmer moving down the lane, "
                   "wall to wall, for every length.",
    "event_underwater": "Please upload a video that shows the whole swim from an underwater camera: the swimmer moving down the lane, "
                        "wall to wall, for every length.",
}


def phase_covered(result: Result) -> bool:
    """True when the module measured at least one thing."""
    return any(value is not None for value in result.metrics.values())


def coverage_message(results: list[Result], phase: str, expected_view: str | None = None, seen_view: str | None = None) -> str | None:
    """A plain message when the video does not match what was chosen, else None.

    phase is a module name or "lap"; expected_view is "above" or "underwater"
    when the coach chose a whole-lap option that says which camera it is.
    """
    if expected_view and seen_view and expected_view != seen_view:
        other = "event" if phase.startswith("event") else "lap"
        if expected_view == "underwater":
            return ("This looks like an above-water video. Please upload an underwater video, "
                    f"or choose 'Whole {other} above water'.")
        return ("This looks like an underwater video. Please upload an above-water video, "
                f"or choose 'Whole {other} under water'.")
    if phase == "lap":
        if not any(phase_covered(r) for r in results):
            return PHASE_MESSAGES["lap"]
        return None
    for r in results:
        if r.module == phase and not phase_covered(r):
            return PHASE_MESSAGES.get(phase, f"Please upload a video that includes the {phase}.")
    return None
