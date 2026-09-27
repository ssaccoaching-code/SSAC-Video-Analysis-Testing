"""Tests for core/ai.py that never touch the network."""

import json
import os
import tempfile
import unittest

from tests.helpers import moving_dot_clip
from core import ai
from core.types import Result


class ParseTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(ai.parse_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self) -> None:
        self.assertEqual(ai.parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_text_around(self) -> None:
        self.assertEqual(ai.parse_json('Here you go: {"a": [1, 2]} thanks'), {"a": [1, 2]})


class EnvTests(unittest.TestCase):
    def test_load_env_file(self) -> None:
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, ".env")
        with open(path, "w") as handle:
            handle.write("# comment\nSPLITSECOND_TEST_KEY = 'abc'\n")
        os.environ.pop("SPLITSECOND_TEST_KEY", None)
        ai.load_env(path)
        self.assertEqual(os.environ.get("SPLITSECOND_TEST_KEY"), "abc")


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clip = moving_dot_clip()
        self.results = [Result(module="dive", metrics={"entry_distance_m": 3.1, "entry_angle_deg": None}, key_frames={"entry": 10})]
        self.frames = {("dive", "entry"): self.clip.frame_at(10)[:, :, ::-1]}

    def test_builds_images_and_text(self) -> None:
        messages = ai.build_messages(self.results, self.frames, "clip", {"course": "SCY", "_private": object()}, "Theo", "felt good", "dive")
        blocks = messages[0]["content"]
        self.assertEqual(sum(1 for b in blocks if b["type"] == "image"), 1)
        text = " ".join(b["text"] for b in blocks if b["type"] == "text")
        self.assertIn("entry_distance_m", text)
        self.assertIn("felt good", text)
        self.assertNotIn("_private", text)

    def test_review_with_fake_api(self) -> None:
        reply = json.dumps({"summary": ["ok"], "plan": [{"focus": "entry", "why": "w", "drill": "d", "target": "t"}],
                            "frame_checks": [{"module": "dive", "label": "entry", "looks_right": True, "comment": "fine"}],
                            "corrections": {}, "report_md": "## Race overview\nshort"})
        original = ai.call_api
        ai.call_api = lambda messages, system=ai.SYSTEM_PROMPT, model=None, max_tokens=6000: reply
        try:
            os.environ["ANTHROPIC_API_KEY"] = "test"
            out = ai.review(self.results, self.frames, "clip", {}, phase="dive")
        finally:
            ai.call_api = original
        self.assertTrue(out.ok)
        self.assertEqual(out.plan[0]["focus"], "entry")
        report = ai.report_with_numbers(out, self.results, "Title")
        self.assertIn("| entry_distance_m | 3.10 |", report)
        self.assertIn("not measured", report)

    def test_no_key_is_a_message_not_a_crash(self) -> None:
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        original = ai.load_env
        ai.load_env = lambda path=None: None
        try:
            out = ai.review(self.results, self.frames, "clip", {})
        finally:
            ai.load_env = original
            if saved:
                os.environ["ANTHROPIC_API_KEY"] = saved
        self.assertFalse(out.ok)
        self.assertIn("API key", out.error)


if __name__ == "__main__":
    unittest.main()
