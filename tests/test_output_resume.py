import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import wra


def message(msg_id: str, timestamp: int, text: str) -> wra.Message:
    return wra.Message(
        id=msg_id,
        timestamp=timestamp,
        time=wra.format_local_time(timestamp),
        sender="我",
        sender_id="self",
        direction="sent",
        type="text",
        text=text,
    )


class OutputAndResumeTests(unittest.TestCase):
    def test_markdown_is_split_and_index_remains_small(self):
        messages = [
            message("1", 1704067200, "甲" * 400),
            message("2", 1706745600, "乙" * 400),
            message("3", 1706745660, "丙" * 400),
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "output") as temp:
            target = Path(temp)
            parts = wra.write_markdown_timeline(target, "测试联系人", messages, max_part_bytes=700)

            index = (target / "timeline.md").read_text(encoding="utf-8")
            self.assertLess((target / "timeline.md").stat().st_size, 4096)
            self.assertEqual(len(parts), 3)
            self.assertTrue(all(path.stat().st_size < 1500 for path in parts))
            self.assertIn("timeline-parts/2024-01.md", index)
            self.assertIn("timeline-parts/2024-02-02.md", index)

    def test_existing_transcripts_are_reused(self):
        payload = {
            "messages": [
                {
                    "id": "voice-1",
                    "timestamp": 1704067200,
                    "transcript": "已经完成的转写",
                    "transcript_status": "ok",
                }
            ]
        }
        messages = [
            wra.Message(
                id="voice-1",
                timestamp=1704067200,
                time="2024-01-01 00:00:00",
                sender="对方",
                sender_id="other",
                direction="received",
                type="voice",
                text="",
                audio_path="voice.silk",
                transcript_status="pending",
            )
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "output") as temp:
            timeline = Path(temp) / "timeline.json"
            timeline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            count = wra.reuse_existing_transcripts(messages, timeline)

        self.assertEqual(count, 1)
        self.assertEqual(messages[0].transcript, "已经完成的转写")
        self.assertEqual(messages[0].transcript_status, "ok")

    def test_silk_decoder_recreates_missing_work_directory(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "output") as temp:
            root = Path(temp)
            source = root / "voice.silk"
            source.write_bytes(b"silk")
            missing = root / "removed" / "nested"

            def fake_decode(_source, wav, rate=24000):
                Path(wav).write_bytes(b"RIFF" + b"0" * 64)

            with patch.dict(sys.modules, {"pilk_nogil": type("Pilk", (), {"silk_to_wav": staticmethod(fake_decode)})()}):
                wav = wra.decode_silk_to_wav(source, missing)

            self.assertTrue(wav.exists())
            self.assertGreater(wav.stat().st_size, 44)


if __name__ == "__main__":
    unittest.main()
