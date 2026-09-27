"""AI review of an analysis with the Claude API.

What it adds on top of the rule-based coach (core/coach.py):
  1. Looks at the key frames (entry, breakout, wall ...) and says whether each
     one really is that moment, with a body-position comment.
  2. Reads every number, the reference bands, the swimmer's saved history and
     any coach or swimmer comments, and writes a summary and a ranked plan.
  3. Writes a technical report (Markdown) the coach can send on.

How to turn it on:
  1. Put the key in a file called .env in the project folder:
         ANTHROPIC_API_KEY=sk-ant-...
     (.env is ignored by git, so the key never reaches GitHub.)
  2. Tick "Review with Claude" in the web page sidebar, or add --ai on the
     command line.

Nothing else in the app depends on this file. If there is no key or no
internet, the app falls back to core/coach.py and says so.

Calls go straight to the HTTPS API with the standard library, so there is no
extra package to install.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.paths import ROOT
from core.types import Result

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
MODELS = ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"]
MAX_IMAGES = 12
IMAGE_MAX_WIDTH = 1024
TIMEOUT_S = 180

BASIS_AI = "Reviewed by Claude ({model}) from the measured numbers, the key frames and the saved history. Still a starting point for the coach."


@dataclass
class AiReview:
    summary: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    plan: list[dict[str, str]] = field(default_factory=list)
    frame_checks: list[dict[str, Any]] = field(default_factory=list)
    corrections: dict[str, Any] = field(default_factory=dict)
    report_md: str = ""
    model: str = ""
    basis: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.summary or self.report_md)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": list(self.summary),
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "plan": [dict(p) for p in self.plan],
            "frame_checks": [dict(c) for c in self.frame_checks],
            "corrections": dict(self.corrections),
            "report_md": self.report_md,
            "model": self.model,
            "basis": self.basis,
            "error": self.error,
        }


# ---------------------------------------------------------------- key and settings

def load_env(path: str | None = None) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing values win)."""
    env_path = path or os.path.join(str(ROOT), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def api_key() -> str | None:
    load_env()
    return os.environ.get("ANTHROPIC_API_KEY") or None


def model_name() -> str:
    load_env()
    return os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL


def available() -> bool:
    return api_key() is not None


# ---------------------------------------------------------------- building the request

def encode_image(image: np.ndarray, rgb: bool = True) -> str:
    """JPEG, base64, at most IMAGE_MAX_WIDTH wide. `rgb` says the array is RGB (page images are)."""
    img = image
    if rgb:
        img = img[:, :, ::-1]
    h, w = img.shape[:2]
    if w > IMAGE_MAX_WIDTH:
        scale = IMAGE_MAX_WIDTH / w
        img = cv2.resize(img, (IMAGE_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode key frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _bands_text() -> str:
    try:
        from core.coach import BANDS
    except Exception:
        return "(no reference bands available)"
    lines = []
    for key, (low, high, direction) in BANDS.items():
        lines.append(f"{key}: good range {low} to {high}, better = {direction}")
    return "\n".join(lines)


def _labels() -> dict[str, str]:
    try:
        from core.report import LABELS
        return dict(LABELS)
    except Exception:
        return {}


def history_text(swimmer: str, limit_runs: int = 6) -> str:
    """The swimmer's last saved runs from the database, compact."""
    if not swimmer:
        return ""
    try:
        from core.db import load_metrics, load_runs
    except Exception:
        return ""
    try:
        runs = [r for r in load_runs() if (r.get("swimmer") or "").strip().lower() == swimmer.strip().lower()][:limit_runs]
        if not runs:
            return ""
        rows = load_metrics()
    except Exception:
        return ""
    by_run: dict[int, list[str]] = {}
    for row in rows:
        if row["run_id"] in {r["id"] for r in runs}:
            value = row["value"]
            if isinstance(value, float):
                value = round(value, 3)
            by_run.setdefault(row["run_id"], []).append(f"{row['module']}.{row['metric']}={value}")
    lines = []
    for run in runs:
        lines.append(f"run {run['id']} on {run.get('run_at', '')} ({run.get('phase', '')}, {run.get('course', '')}, clip {run.get('clip', '')}): "
                     + "; ".join(by_run.get(run["id"], [])))
    return "\n".join(lines)


def _results_payload(results: list[Result], clip_info: str, options: dict[str, Any]) -> dict[str, Any]:
    public = {k: v for k, v in options.items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    return {"clip": clip_info, "options": public, "results": [r.to_dict() for r in results]}


SYSTEM_PROMPT = """You are a competitive swimming technique analyst working with a club coach.
You get: the numbers a computer-vision pipeline measured from one race video, the reference bands the coach uses,
labelled key frames (still images) the pipeline picked, the swimmer's saved history when there is one, and any comments
from the coach or the swimmer.

Your job:
1. Check each key frame: is it really the moment its label says (entry, breakout, wall contact, ...)? Describe the
   body position in one sentence. Where the image lets you, estimate entry angle, depth, streamline quality, kick.
2. Judge every measured number against the bands and against the history. Flag numbers that look like measurement
   errors (impossible values, inconsistent with the frames) rather than swimmer problems.
3. Write a clear summary, the strengths, the weaknesses, and a ranked improvement plan with a concrete drill and a
   numeric target for each item. Most important item first. At most 5 items.
4. Write a technical report in Markdown for the coach: headings Race overview, Phase by phase (dive, underwater,
   swim, turn as available), Trend versus history (or 'no history yet'), Measurement quality, Priorities.
   Plain sentences, numbers with units, no filler.

Units: metres, seconds, degrees, strokes per minute. Say "not measured" for null values; never invent a number.
Answer with ONE JSON object and nothing else, with exactly these keys:
{
  "summary": ["sentence", ...],
  "strengths": ["...", ...],
  "weaknesses": ["...", ...],
  "plan": [{"focus": "...", "why": "...", "drill": "...", "target": "..."}, ...],
  "frame_checks": [{"module": "...", "label": "...", "looks_right": true, "comment": "...", "estimates": {"entry_angle_deg": 38}}, ...],
  "corrections": {"module.metric": value, ...},
  "report_md": "..."
}
"corrections" holds only numbers you are confident are wrong and can replace from what you see; usually it is empty."""


def build_messages(results: list[Result], key_frames: dict[tuple[str, str], np.ndarray], clip_info: str,
                   options: dict[str, Any], swimmer: str = "", comments: str = "", phase: str = "") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    labels = _labels()
    text = [f"Phase analyzed: {phase or 'unknown'}", f"Swimmer: {swimmer or 'not given'}"]
    if comments.strip():
        text.append("Comments from coach or swimmer:\n" + comments.strip())
    text.append("Metric labels (key: plain name):\n" + "\n".join(f"{k}: {v}" for k, v in labels.items()))
    text.append("Reference bands:\n" + _bands_text())
    hist = history_text(swimmer)
    text.append("Saved history for this swimmer:\n" + (hist or "none"))
    text.append("Measured results (JSON):\n" + json.dumps(_results_payload(results, clip_info, options), indent=1, default=str))
    content.append({"type": "text", "text": "\n\n".join(text)})

    items = list(key_frames.items())[:MAX_IMAGES]
    if items:
        content.append({"type": "text", "text": f"Key frames ({len(items)}), each preceded by its module and label:"})
        for (module, label), image in items:
            try:
                data = encode_image(image, rgb=True)
            except Exception:
                continue
            content.append({"type": "text", "text": f"module={module} label={label}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------- calling the API

def call_api(messages: list[dict[str, Any]], system: str = SYSTEM_PROMPT, model: str | None = None,
             max_tokens: int = 6000) -> str:
    key = api_key()
    if not key:
        raise RuntimeError("no ANTHROPIC_API_KEY (put it in .env)")
    body = json.dumps({"model": model or model_name(), "max_tokens": max_tokens, "system": system, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-api-key": key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Claude API error {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach the Claude API: {error.reason}") from None
    parts = [block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"]
    return "".join(parts)


def parse_json(text: str) -> dict[str, Any]:
    """The reply should be one JSON object; tolerate code fences or stray text around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def review(results: list[Result], key_frames: dict[tuple[str, str], np.ndarray] | None = None, clip_info: str = "",
           options: dict[str, Any] | None = None, swimmer: str = "", comments: str = "", phase: str = "",
           model: str | None = None) -> AiReview:
    """Ask Claude to review one analysis. Never raises: check .error / .ok."""
    model = model or model_name()
    out = AiReview(model=model, basis=BASIS_AI.format(model=model))
    if not available():
        out.error = "No API key. Put ANTHROPIC_API_KEY=... in a .env file in the project folder."
        return out
    try:
        messages = build_messages(results, key_frames or {}, clip_info, options or {}, swimmer, comments, phase)
        raw = call_api(messages, model=model)
        data = parse_json(raw)
    except Exception as error:
        out.error = str(error)
        return out
    out.summary = [str(s) for s in data.get("summary", [])]
    out.strengths = [str(s) for s in data.get("strengths", [])]
    out.weaknesses = [str(s) for s in data.get("weaknesses", [])]
    out.plan = [{k: str(p.get(k, "")) for k in ("focus", "why", "drill", "target")} for p in data.get("plan", []) if isinstance(p, dict)]
    out.frame_checks = [c for c in data.get("frame_checks", []) if isinstance(c, dict)]
    out.corrections = data.get("corrections", {}) if isinstance(data.get("corrections"), dict) else {}
    out.report_md = str(data.get("report_md", ""))
    return out


def report_with_numbers(review_out: AiReview, results: list[Result], title: str) -> str:
    """The Markdown report plus a numbers appendix, ready to save or send."""
    labels = _labels()
    lines = [f"# {title}", "", review_out.report_md.strip(), "", "## Numbers", ""]
    for result in results:
        lines.append(f"### {result.module.replace('_', ' ')}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, value in result.metrics.items():
            shown = "not measured" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))
            lines.append(f"| {labels.get(key, key)} | {shown} |")
        lines.append("")
    if review_out.frame_checks:
        lines += ["## Key frame checks", "", "| Module | Frame | Looks right | Comment |", "|---|---|---|---|"]
        for check in review_out.frame_checks:
            lines.append(f"| {check.get('module', '')} | {check.get('label', '')} | {'yes' if check.get('looks_right') else 'no'} | {check.get('comment', '')} |")
        lines.append("")
    lines.append(f"_{review_out.basis}_")
    return "\n".join(lines)
