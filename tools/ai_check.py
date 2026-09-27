"""Check that the AI review works: key found, API reachable, reply parses.

    python tools/ai_check.py

Uses the cartoon clip and made-up numbers, so it costs a fraction of a cent.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ai  # noqa: E402
from core.types import Result  # noqa: E402
from core.video import open_clip  # noqa: E402


def main() -> int:
    provider = ai.provider_name()
    if provider is None:
        print("No key found. Put OPENAI_API_KEY=... or ANTHROPIC_API_KEY=... in .env next to run.py")
        return 1
    print(f"provider: {provider}   model: {ai.model_name()}")
    sample = Path("samples/synthetic_sprint.mp4")
    if not sample.exists():
        from tools.make_sample_clip import make_clip
        make_clip(str(sample))
    with open_clip(str(sample)) as clip:
        frames = {("dive", "entry"): clip.frame_at(45)[:, :, ::-1], ("underwater", "breakout"): clip.frame_at(120)[:, :, ::-1]}
        info = str(clip)
    results = [
        Result(module="dive", metrics={"reaction_time_s": 0.68, "flight_time_s": 0.41, "entry_distance_m": 3.1,
                                       "entry_angle_deg": None, "entry_depth_m": None}, key_frames={"entry": 45}),
        Result(module="underwater", metrics={"underwater_distance_m": 7.8, "underwater_time_s": 3.9, "kick_count": 6,
                                             "breakout_time_s": 4.6, "breakout_speed_mps": 1.7, "breakout_slowdown_pct": 24.0},
               key_frames={"breakout": 120}),
    ]
    started = time.time()
    out = ai.review(results, frames, clip_info=info, options={"course": "SCY", "start_s": 1.0}, swimmer="Test",
                    comments="This is a cartoon test clip, not a real swimmer.", phase="lap")
    print(f"took {time.time() - started:.1f} s")
    if not out.ok:
        print(f"FAILED: {out.error}")
        return 1
    print("OK. Summary:")
    for line in out.summary:
        print(f"  {line}")
    print(f"plan items: {len(out.plan)}   frame checks: {len(out.frame_checks)}   report chars: {len(out.report_md)}")
    os.makedirs("results", exist_ok=True)
    with open("results/ai_check_report.md", "w", encoding="utf-8") as handle:
        handle.write(ai.report_with_numbers(out, results, "AI check"))
    print("saved results/ai_check_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
