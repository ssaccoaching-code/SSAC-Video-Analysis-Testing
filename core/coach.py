"""A plain-language summary and a tentative improvement plan from the numbers.

    from core.coach import coach
    advice = coach(results)
    advice.summary   list of sentences: what was measured, what looked strong, what looked weak
    advice.plan      list of dicts: focus, why, drill, target  (most important first)
    advice.basis     one line saying where the reference ranges come from

Everything here is rule based: each metric is compared with a reference band
for club-level age-group and high-school swimmers and the results are turned
into sentences. No model, no internet. The bands live in BANDS below so a
coach can move them. The plan is a starting point for the coach, not a
prescription, and it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.types import Result

BASIS = ("Tentative: reference ranges are general club-level age-group and high-school targets, "
         "not this swimmer's own history. The coach decides what to keep.")

# metric -> (good_low, good_high, direction) where direction says which way is better:
# "low" = smaller is better, "high" = larger is better, "band" = inside the band is right.
BANDS: dict[str, tuple[float, float, str]] = {
    "reaction_time_s": (0.55, 0.75, "low"),
    "flight_time_s": (0.30, 0.55, "band"),
    "entry_distance_m": (2.8, 4.0, "high"),
    "entry_angle_deg": (30.0, 45.0, "band"),
    "entry_depth_m": (0.5, 1.1, "band"),
    "underwater_distance_m": (5.0, 12.0, "high"),
    "breakout_slowdown_pct": (0.0, 20.0, "low"),
    "turn_time_s": (4.5, 6.5, "low"),
    "wall_contact_s": (0.0, 0.45, "low"),
    "push_off_speed_mps": (2.3, 3.2, "high"),
    "distance_per_stroke_m": (1.4, 2.4, "high"),
    "tempo_change_pct": (-6.0, 6.0, "band"),
    "speed_change_pct": (-6.0, 4.0, "band"),
    "strokes_change_pct": (-8.0, 10.0, "band"),
    "underwater_change_pct": (-25.0, 15.0, "band"),
}

UNDERWATER_LIMIT_M = 15.0


@dataclass
class Advice:
    summary: list[str] = field(default_factory=list)
    plan: list[dict[str, str]] = field(default_factory=list)
    basis: str = BASIS

    def to_dict(self) -> dict[str, Any]:
        return {"summary": list(self.summary), "plan": [dict(p) for p in self.plan], "basis": self.basis}


@dataclass
class _Finding:
    severity: float          # bigger = more important; 0 = a strength
    text: str                # for the summary
    focus: str = ""
    why: str = ""
    drill: str = ""
    target: str = ""


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _flatten(results: list[Result]) -> tuple[dict[str, Any], list[str], list[str]]:
    """All metrics in one dict (top-level results win over details), module names, notes."""
    metrics: dict[str, Any] = {}
    modules: list[str] = []
    notes: list[str] = []
    for r in results:
        modules.append(r.module)
        notes.extend(r.notes)
        for d in r.details:
            for key, value in d.metrics.items():
                if value is not None and key not in metrics:
                    metrics[key] = value
            notes.extend(d.notes)
    for r in results:
        for key, value in r.metrics.items():
            if value is not None:
                metrics[key] = value
    return metrics, modules, notes


def _where(value: float, key: str) -> str:
    """'good', 'ok' or 'poor' against the band."""
    lo, hi, direction = BANDS[key]
    if direction == "low":
        return "good" if value <= lo else ("ok" if value <= hi else "poor")
    if direction == "high":
        return "good" if value >= hi else ("ok" if value >= lo else "poor")
    if lo <= value <= hi:
        return "good"
    slack = 0.35 * (hi - lo)
    return "ok" if lo - slack <= value <= hi + slack else "poor"


def _list_stats(value: Any) -> tuple[list[float], float | None, float | None]:
    if not isinstance(value, (list, tuple)):
        return [], None, None
    good = [float(v) for v in value if _num(v) is not None]
    if not good:
        return [], None, None
    return good, min(good), max(good)


def coach(results: list[Result]) -> Advice:
    advice = Advice()
    m, modules, notes = _flatten(results)
    if not m:
        advice.summary.append("Nothing was measured in this clip, so there is no summary or plan.")
        return advice
    rough = any("guessed scale" in n or "no calibration" in n for n in notes)
    findings: list[_Finding] = []

    # ------------------------------------------------------------ the start
    v = _num(m.get("reaction_time_s"))
    if v is not None:
        w = _where(v, "reaction_time_s")
        if w == "good":
            findings.append(_Finding(0, f"Reaction off the block was quick ({v:.2f} s)."))
        elif w == "poor":
            findings.append(_Finding(3, f"Reaction time was slow ({v:.2f} s; aim for under 0.70 s).",
                                     "Start reaction", f"{v:.2f} s from the signal to leaving the block",
                                     "Reaction starts to a whistle, 6 to 8 per session: weight forward, hands pulling on the block, "
                                     "go on the sound not the sight of others",
                                     "under 0.70 s"))
    v = _num(m.get("entry_angle_deg"))
    if v is not None:
        w = _where(v, "entry_angle_deg")
        if w == "good":
            findings.append(_Finding(0, f"Entry angle of {v:.0f} degrees is in the right range."))
        elif v < BANDS["entry_angle_deg"][0]:
            findings.append(_Finding(2.5, f"Entry was flat ({v:.0f} degrees), which costs speed at the splash.",
                                     "Steeper, cleaner entry", f"{v:.0f} degree entry, flatter than the 30 to 45 degree target",
                                     "Dives over a noodle held 1 m out, then through a hoop: hands, head, hips and feet through one hole",
                                     "30 to 45 degrees, one small splash"))
        else:
            findings.append(_Finding(2.0, f"Entry was steep ({v:.0f} degrees), which sends the swimmer deep.",
                                     "Flatter entry", f"{v:.0f} degree entry, steeper than the 30 to 45 degree target",
                                     "Dive for distance over a lane rope 3 m out; look at the far wall, not the bottom, on the flight",
                                     "30 to 45 degrees, breakout inside 10 m"))
    v = _num(m.get("entry_distance_m"))
    if v is not None and not rough:
        w = _where(v, "entry_distance_m")
        if w == "good":
            findings.append(_Finding(0, f"Good distance off the block ({v:.1f} m to the entry)."))
        elif w == "poor":
            findings.append(_Finding(2.0, f"The dive was short ({v:.1f} m to the entry).",
                                     "Drive off the block", f"{v:.1f} m entry distance, under the 2.8 m target",
                                     "Block jumps for distance onto a mat, then dives to a rope at 3 m: full leg drive, arms thrown forward",
                                     "3 m or more"))
    v = _num(m.get("entry_depth_m"))
    if v is not None and _where(v, "entry_depth_m") == "poor" and v > BANDS["entry_depth_m"][1]:
        findings.append(_Finding(2.0, f"The entry went deep ({v:.1f} m), which wastes the dive speed.",
                                 "Shallower entry", f"{v:.1f} m deep after the entry",
                                 "Steer up with the hands right after the entry; dive over a rope with a 'skim the surface' cue",
                                 "0.5 to 1.0 m"))

    # ------------------------------------------------------------ underwater
    uw_list, uw_min, uw_max = _list_stats(m.get("underwater_distance_by_length_m"))
    v = _num(m.get("underwater_distance_m")) if not uw_list else uw_min
    if v is not None and not rough:
        if uw_max is not None and uw_max > UNDERWATER_LIMIT_M:
            findings.append(_Finding(4, f"One breakout was past the 15 m mark ({uw_max:.1f} m); that is a disqualification in fly, back and free.",
                                     "Surface before 15 m", f"breakout at {uw_max:.1f} m",
                                     "Count kicks to the flags and break out on a fixed kick number; watch the 15 m mark on every push-off",
                                     "breakout between 10 and 14 m"))
        w = _where(v, "underwater_distance_m")
        label = "underwater distance" if not uw_list else "shortest underwater"
        if w == "good":
            findings.append(_Finding(0, f"Strong underwaters ({label} {v:.1f} m)."))
        elif w == "poor":
            findings.append(_Finding(3, f"The {label} was short ({v:.1f} m before the breakout).",
                                     "Longer underwaters", f"{label} {v:.1f} m, under the 5 m target",
                                     "Streamline kick sets: 4 x 25 with a set kick count off every wall, add one kick each week; "
                                     "vertical kicking 8 x 15 s for kick power",
                                     "5 to 10 m on every wall"))
    v = _num(m.get("breakout_slowdown_pct"))
    slow_list, _, slow_max = _list_stats(m.get("breakout_slowdown_by_length_pct"))
    if slow_max is not None:
        v = slow_max
    if v is not None:
        w = _where(v, "breakout_slowdown_pct")
        if w == "good":
            findings.append(_Finding(0, f"Breakouts came while the underwater speed was still up ({v:.0f}% slowdown)."))
        elif w == "poor":
            findings.append(_Finding(3, f"The swimmer slowed {v:.0f}% before breaking out, which means the glide went on too long.",
                                     "Break out earlier", f"{v:.0f}% slowdown from peak underwater speed to the breakout",
                                     "Breakout drill: 6 x 25 from a push, break out one kick earlier each rep and note where the speed feels highest; "
                                     "first stroke starts before the head is fully up",
                                     "under 20% slowdown"))
    kicks_list, k_min, k_max = _list_stats(m.get("kicks_by_length"))
    if len(kicks_list) >= 2 and k_max and k_min is not None and k_max - k_min >= 3:
        findings.append(_Finding(2.0, f"Kick counts were uneven across the walls ({k_min:.0f} to {k_max:.0f} kicks).",
                                 "Same kick count off every wall", f"{k_min:.0f} to {k_max:.0f} kicks per wall",
                                 "Pick a kick number and hold it on every push-off in a 8 x 50 set; count out loud on the deck",
                                 "within one kick of the plan on every wall"))
    uw_change = _num(m.get("underwater_change_pct"))
    if uw_change is not None and _where(uw_change, "underwater_change_pct") == "poor" and uw_change < 0:
        findings.append(_Finding(2.5, f"Underwaters got shorter as the race went on ({uw_change:.0f}% by the last length).",
                                 "Hold the underwaters late in the race", f"{uw_change:.0f}% shorter breakout on the last length",
                                 "Descending 4 x 100 with the same breakout distance on every wall; the last wall counts most",
                                 "last length breakout within 25% of the second length"))

    # ------------------------------------------------------------ the swim
    v = _num(m.get("tempo_change_pct"))
    if v is not None:
        w = _where(v, "tempo_change_pct")
        if w == "good":
            findings.append(_Finding(0, f"Tempo held steady ({v:+.0f}% over the swim)."))
        elif v < 0:
            findings.append(_Finding(3, f"Tempo dropped {abs(v):.0f}% over the swim.",
                                     "Hold tempo to the wall", f"{v:+.0f}% tempo from early to late",
                                     "Tempo trainer sets: 8 x 50 at race tempo with the beeper, then 4 x 100 holding the same setting; "
                                     "build kick in the last 10 m",
                                     "tempo within 5% of the first length"))
        else:
            findings.append(_Finding(2.0, f"Tempo rose {v:.0f}% late in the swim while the stroke got shorter.",
                                     "Hold stroke length under fatigue", f"{v:+.0f}% tempo with less distance per stroke",
                                     "Stroke count sets: 6 x 100 holding the stroke count of the first 25; long, finished pulls",
                                     "stroke count within 2 of the first length"))
    v = _num(m.get("strokes_change_pct"))
    if v is not None and _where(v, "strokes_change_pct") == "poor" and v > 0:
        findings.append(_Finding(2.5, f"Stroke count per length climbed {v:.0f}% by the end, so each stroke moved the swimmer less.",
                                 "Distance per stroke late in the race", f"{v:+.0f}% strokes on the last length",
                                 "Stroke count golf: 8 x 50 lowering time plus strokes; catch-up and fist drills for a longer pull",
                                 "stroke count within 10% of the second length"))
    v = _num(m.get("distance_per_stroke_m"))
    if v is not None and not rough:
        w = _where(v, "distance_per_stroke_m")
        if w == "good":
            findings.append(_Finding(0, f"Long stroke ({v:.2f} m per stroke)."))
        elif w == "poor":
            findings.append(_Finding(2.0, f"Short stroke ({v:.2f} m per stroke).",
                                     "Stroke length", f"{v:.2f} m per stroke, under the 1.4 m target",
                                     "Fist drill and single-arm drill 4 x 50 each, then 4 x 50 counting strokes; finish every pull past the hip",
                                     "1.5 m or more per stroke"))

    # ------------------------------------------------------------ turns
    tt_list, tt_min, tt_max = _list_stats(m.get("turn_time_by_turn_s"))
    v = _num(m.get("turn_time_avg_s")) if tt_list else _num(m.get("turn_time_s"))
    if v is not None and not rough:
        w = _where(v, "turn_time_s")
        zone = _num(m.get("turn_zone_m"))
        zone_text = f"{zone:g} m in / {zone:g} m out" if zone else "5 m in / 5 m out"
        if w == "good":
            findings.append(_Finding(0, f"Fast turns ({v:.1f} s {zone_text})."))
        elif w == "poor":
            findings.append(_Finding(3, f"Turns were slow ({v:.1f} s {zone_text}).",
                                     "Faster turns", f"{v:.1f} s for {zone_text}, over the 6.5 s target",
                                     "Turn sets: 12 x 50 with a fast last 5 m in, tight tuck, feet planted and straight off; "
                                     "time each turn flags to flags",
                                     "under 6 s flags to flags"))
    if tt_list and len(tt_list) >= 2 and tt_max is not None and tt_min is not None and tt_max - tt_min > 1.0:
        findings.append(_Finding(2.0, f"Turns were uneven ({tt_min:.1f} to {tt_max:.1f} s).",
                                 "Consistent turns", f"{tt_max - tt_min:.1f} s between the fastest and slowest turn",
                                 "Every turn in practice at race speed, no easy turns; count strokes into each wall",
                                 "all turns within 0.5 s of each other"))
    wc_list, _, wc_max = _list_stats(m.get("wall_contact_by_turn_s"))
    v = wc_max if wc_max is not None else _num(m.get("wall_contact_s"))
    turn_type = str(m.get("turn_type") or "")
    if v is not None:
        limit = 0.45 if turn_type != "open" else 0.8
        if v > limit:
            findings.append(_Finding(2.5, f"Time on the wall was long ({v:.2f} s).",
                                     "Quicker off the wall", f"{v:.2f} s on the wall",
                                     "Wall touch-and-go drill: 10 x (swim in, touch, push) with a partner calling 'go' on contact; "
                                     "for flip turns, plant and push in one motion",
                                     f"under {limit:.2f} s"))
    po_list, po_min, _ = _list_stats(m.get("push_off_speed_by_turn_mps"))
    v = po_min if po_min is not None else _num(m.get("push_off_speed_mps"))
    if v is not None and not rough:
        w = _where(v, "push_off_speed_mps")
        if w == "good":
            findings.append(_Finding(0, f"Strong push-offs ({v:.1f} m/s)."))
        elif w == "poor":
            findings.append(_Finding(2.5, f"Push-off speed was weak ({v:.1f} m/s).",
                                     "Push-off power", f"{v:.1f} m/s leaving the wall",
                                     "Streamline push-offs for distance, 10 per session, with a tight streamline and no stroke until the flags; "
                                     "squat jumps on deck",
                                     "2.5 m/s or more"))

    # ------------------------------------------------------------ the whole event
    v = _num(m.get("speed_change_pct"))
    splits, sp_min, sp_max = _list_stats(m.get("split_by_length_s"))
    if v is not None:
        w = _where(v, "speed_change_pct")
        if w == "good":
            findings.append(_Finding(0, f"Pacing was even ({v:+.0f}% speed from the second length to the last)."))
        elif v < 0:
            findings.append(_Finding(3.5, f"The swimmer faded: the last length was {abs(v):.0f}% slower than the second.",
                                     "Pacing and finish", f"{v:+.0f}% speed on the last length",
                                     "Broken swims of the race distance with 10 s rest at each wall, holding the goal split on every length; "
                                     "hold back 2 to 3% on the first length in practice races",
                                     "last length within 5% of the second"))
        else:
            findings.append(_Finding(1.5, f"The swimmer finished faster than the early lengths ({v:+.0f}%), which may mean the start was too easy.",
                                     "Get out faster", f"{v:+.0f}% speed on the last length",
                                     "Race-pace 50s from a dive at goal split; practice the first length at race effort with a controlled breakout",
                                     "first two lengths within 3% of the goal split"))
    if len(splits) >= 3 and sp_min and sp_max and (sp_max - sp_min) / sp_min > 0.12:
        findings.append(_Finding(2.0, f"Splits varied a lot ({sp_min:.1f} to {sp_max:.1f} s per length)."))

    # ------------------------------------------------------------ write it up
    what = _describe(results, m)
    if what:
        advice.summary.append(what)
    strengths = [f.text for f in findings if f.severity == 0]
    problems = sorted([f for f in findings if f.severity > 0], key=lambda f: -f.severity)
    if strengths:
        advice.summary.append("Looked good: " + " ".join(strengths[:4]))
    if problems:
        advice.summary.append("To work on: " + " ".join(p.text for p in problems[:4]))
    if not strengths and not problems:
        advice.summary.append("The numbers are inside the usual ranges; nothing stood out either way.")
    if rough:
        advice.summary.append("Distances and speeds in this clip use a guessed scale, so the plan only looks at times, "
                              "angles and percentages. Calibrate the camera for the rest.")
    seen_focus: set[str] = set()
    for p in problems:
        if not p.focus or p.focus in seen_focus:
            continue
        seen_focus.add(p.focus)
        advice.plan.append({"focus": p.focus, "why": p.why, "drill": p.drill, "target": p.target})
        if len(advice.plan) >= 5:
            break
    if not advice.plan and (strengths or problems):
        advice.plan.append({"focus": "Keep it up", "why": "nothing measured fell outside the reference ranges",
                            "drill": "Race-pace sets at the current numbers; re-film in two weeks and compare",
                            "target": "same numbers or better under fatigue"})
    return advice


def _describe(results: list[Result], m: dict[str, Any]) -> str:
    names = [r.module for r in results]
    if any(n.startswith("event") for n in names):
        lengths = m.get("lengths")
        dist = m.get("distance_m")
        t = m.get("event_time_s")
        speed = m.get("avg_speed_mps")
        parts = []
        if lengths:
            parts.append(f"{lengths} length{'s' if lengths != 1 else ''}")
        if dist:
            parts.append(f"{dist:.0f} m")
        if t:
            parts.append(f"{t:.2f} s")
        if speed:
            parts.append(f"{speed:.2f} m/s average")
        return "Whole event: " + ", ".join(parts) + "." if parts else "Whole event analysed."
    measured = [n.replace("_", " ") for n in names if any(v is not None for v in next((r.metrics for r in results if r.module == n), {}).values())]
    if not measured:
        return ""
    return "Measured: " + ", ".join(measured) + "."
