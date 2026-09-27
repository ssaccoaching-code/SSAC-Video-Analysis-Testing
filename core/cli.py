"""Shared command line handling.

run.py (all modules) and every modules/<name>/cli.py (one module) use this,
so they all take the same flags.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from core.base import AnalysisModule
from core.report import coverage_message, to_json, to_text
from core.types import Result
from core.video import open_clip


def parse_option(text: str) -> tuple[str, Any]:
    """Turn 'key=value' into a typed (key, value) pair."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got '{text}'")
    key, raw = text.split("=", 1)
    for convert in (int, float):
        try:
            return key, convert(raw)
        except ValueError:
            pass
    return key, raw


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("clip", nargs="?", help="path to a video file")
    parser.add_argument("--start", type=float, default=0.0, help="time of the start signal in the clip, seconds")
    parser.add_argument("--course", default="SCY", choices=["SCY", "SCM", "LCM"], help="pool course (default SCY)")
    parser.add_argument("--lane", type=int, default=None, help="lane of the swimmer of interest")
    parser.add_argument("--option", "-o", action="append", default=[], type=parse_option, metavar="KEY=VALUE",
                        help="extra option passed to the modules; repeatable")
    parser.add_argument("--json", metavar="FILE", help="also write the results as JSON to FILE")
    parser.add_argument("--ai", action="store_true", help="also ask the AI (ChatGPT or Claude, from .env) to review the numbers and key frames")
    parser.add_argument("--swimmer", default="", help="swimmer name, used to pull saved history for the AI review")
    parser.add_argument("--comments", default="", help="coach or swimmer comments for the AI review")
    parser.add_argument("--report", metavar="FILE", help="write the AI technical report (Markdown) to FILE")


def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options = {"start_s": args.start, "course": args.course, "lane": args.lane}
    options.update(dict(args.option))
    if getattr(args, "ai", False):
        options["_ai"] = {"swimmer": args.swimmer, "comments": args.comments, "report": args.report}
    return options


def ai_review_cli(results: list[Result], clip_path: str, options: dict[str, Any], phase: str) -> None:
    """Print the Claude review and save the report, when --ai was given."""
    settings = options.get("_ai")
    if not settings:
        return
    from core import ai

    with open_clip(clip_path) as clip:
        frames = {(r.module, label): clip.frame_at(index)[:, :, ::-1]
                  for top in results for r in [top] + list(top.details) for label, index in r.key_frames.items()}
        clip_info = str(clip)
    print(f"asking {ai.provider_name()} ({ai.model_name()})...")
    out = ai.review(results, frames, clip_info=clip_info, options=options, swimmer=settings.get("swimmer", ""),
                    comments=settings.get("comments", ""), phase=phase)
    if not out.ok:
        print(f"AI review not available: {out.error}")
        return
    print("[AI review]")
    for sentence in out.summary:
        print(f"  {sentence}")
    for item in out.plan:
        print(f"  plan: {item['focus']}: {item['drill']} (target {item['target']})")
    for check in out.frame_checks:
        flag = "ok " if check.get("looks_right") else "?? "
        print(f"  frame {flag}{check.get('module')}/{check.get('label')}: {check.get('comment', '')}")
    report_path = settings.get("report")
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(ai.report_with_numbers(out, results, f"SSAC SplitSecond: {os.path.basename(clip_path)}"))
        print(f"saved {report_path}")


def run_modules(modules: list[AnalysisModule], clip_path: str, options: dict[str, Any], json_path: str | None = None) -> list[Result]:
    results: list[Result] = []
    with open_clip(clip_path) as clip:
        print(f"clip: {clip}")
        for module in modules:
            results.append(module.run(clip, options))
    print(to_text(results))
    for module in modules:
        message = coverage_message(results, module.name)
        if message:
            print(f"{module.name}: {message}")
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write(to_json(results, clip_path=clip_path))
        print(f"saved {json_path}")
    ai_review_cli(results, clip_path, options, ",".join(m.name for m in modules))
    return results


def single_module_main(module: AnalysisModule, argv: list[str] | None = None) -> int:
    """The whole main() for a modules/<name>/cli.py script."""
    parser = argparse.ArgumentParser(description=f"{module.name}: {module.description}")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    if not args.clip:
        parser.error("need a clip path")
    run_modules([module], args.clip, options_from_args(args), args.json)
    return 0


def run_lap(clip_path: str, options: dict[str, Any], json_path: str | None = None) -> list[Result]:
    """Whole-lap run: every phase in order on one clip (see core/lap.py)."""
    from core.lap import analyze_lap

    with open_clip(clip_path) as clip:
        print(f"clip: {clip}")
        results = analyze_lap(clip, options)
    print(to_text(results))
    message = coverage_message(results, "lap")
    if message:
        print(message)
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write(to_json(results, clip_path=clip_path))
        print(f"saved {json_path}")
    ai_review_cli(results, clip_path, options, "lap")
    return results
