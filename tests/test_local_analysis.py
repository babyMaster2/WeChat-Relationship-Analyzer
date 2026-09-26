import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import local_analysis
import wra


class SenderInferenceTests(unittest.TestCase):
    def test_xml_direction_repairs_contaminated_sender_names(self):
        rows = [
            {"sender_id": 1, "isSend": 0, "senderDisplayName": "错误甲", "content": '<msg fromusername="wxid_contact" />'},
            {"sender_id": 3, "isSend": 0, "senderDisplayName": "错误乙", "content": '<msg><emoji fromusername = "wxid_self" tousername="wxid_contact" /></msg>'},
            {"sender_id": 3, "isSend": 0, "senderDisplayName": "错误乙", "content": "普通文字"},
        ]
        roles = wra.infer_sender_roles(rows, {"session": {"id": "wxid_contact"}})
        self.assertEqual(roles, {"1": "received", "3": "sent"})

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "messages.json"
            messages = [wra.normalize_message(row, path, {}, [], "我", roles) for row in rows]
        self.assertEqual([message.sender for message in messages], ["对方", "我", "我"])


class LocalAnalysisTests(unittest.TestCase):
    def test_cloud_runtime_is_rejected(self):
        with self.assertRaises(RuntimeError):
            local_analysis.validate_local_config({"analysis_backend": "https://example.com"})

    def test_prompt_contains_required_evidence_rules(self):
        timeline = {
            "session": "测试联系人",
            "messages": [
                {"id": "1", "timestamp": 1700000000, "time": "2023-11-15 06:13:20", "sender": "我", "direction": "sent", "type": "text", "text": "晚安"},
                {"id": "2", "timestamp": 1700000060, "time": "2023-11-15 06:14:20", "sender": "对方", "direction": "received", "type": "text", "text": "你也早点睡"},
            ],
        }
        evidence = local_analysis.build_evidence_pack(timeline, max_chars=10000)
        prompt = local_analysis.build_prompt("测试联系人", evidence)
        self.assertIn("事实 / 高可信推断 / 可能解释 / 无法判断", prompt)
        self.assertIn("2023-11-15 06:13:20｜我：晚安", prompt)
        self.assertNotIn("API", prompt)

    def test_analyzer_uses_prompt_file_not_command_line(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-cli.exe"
            model = root / "model.gguf"
            timeline = root / "timeline.json"
            runtime.write_bytes(b"exe")
            model.write_bytes(b"gguf")
            timeline.write_text(json.dumps({"session": "甲", "messages": []}, ensure_ascii=False), encoding="utf-8")
            config = {
                "analysis_backend": "local",
                "local_llm_runtime": str(runtime),
                "local_llm_model": str(model),
                "analysis_context": 4096,
                "analysis_max_tokens": 100,
                "analysis_threads": 2,
            }
            completed = type("Result", (), {"returncode": 0, "stdout": "# 关系分析报告\n\n## 证据台账\n当前测试证据不足，无法判断。", "stderr": ""})()
            with patch("local_analysis.subprocess.run", return_value=completed) as run:
                report = local_analysis.analyze_timeline(timeline, config, output_name="relationship-report.local-test.md")
            args = run.call_args.args[0]
            self.assertIn("-f", args)
            self.assertFalse(any("关系分析报告" in str(arg) for arg in args))
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
