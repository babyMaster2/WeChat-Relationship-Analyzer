import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import wra


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "output")
        self.config = {"model": "medium", "model_cache_dir": self.temp.name}

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_model_cache_stays_beside_project(self):
        cache = wra.resolve_model_cache_dir({"model_cache_dir": "../models/faster-whisper"})
        self.assertEqual(cache, (PROJECT_ROOT.parent / "models" / "faster-whisper").resolve())

    def test_prepare_model_announces_download_and_uses_configured_cache(self):
        config = self.config
        expected = str(Path(self.temp.name) / "snapshot")
        output = io.StringIO()
        with patch.object(wra, "download_model_from_modelscope", return_value=Path(expected)) as download:
            with redirect_stdout(output):
                actual = wra.prepare_model(config)

        self.assertEqual(actual, Path(expected))
        self.assertIn("首次准备语音模型", output.getvalue())
        self.assertEqual(download.call_args.args[1], wra.resolve_model_cache_dir(config) / "medium")

    def test_prepare_model_falls_back_when_official_endpoint_times_out(self):
        config = self.config
        expected = str(Path(self.temp.name) / "snapshot")
        output = io.StringIO()
        with patch.object(wra, "download_model_from_modelscope", side_effect=TimeoutError("domestic")):
            with patch("huggingface_hub.snapshot_download", side_effect=TimeoutError("timeout")) as download:
                with patch.object(wra, "download_model_from_mirror", return_value=Path(expected)) as mirror:
                    with redirect_stdout(output):
                        actual = wra.prepare_model(config)

        self.assertEqual(actual, Path(expected))
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.kwargs["endpoint"], "https://huggingface.co")
        mirror.assert_called_once()
        self.assertIn("最后备用线路", output.getvalue())

    def test_prepare_model_retries_transient_domestic_failure(self):
        config = self.config
        expected = Path(self.temp.name) / "snapshot"
        with patch.object(wra, "download_model_from_modelscope", side_effect=[TimeoutError("handshake"), expected]) as domestic:
            with patch("huggingface_hub.snapshot_download") as huggingface:
                actual = wra.prepare_model(config)

        self.assertEqual(actual, expected)
        self.assertEqual(domestic.call_count, 2)
        huggingface.assert_not_called()

    def test_prepare_model_prefers_modelscope_for_medium(self):
        config = self.config
        expected = Path(self.temp.name) / "medium"
        with patch.object(wra, "download_model_from_modelscope", return_value=expected) as domestic:
            with patch("huggingface_hub.snapshot_download") as huggingface:
                actual = wra.prepare_model(config)

        self.assertEqual(actual, expected)
        domestic.assert_called_once()
        huggingface.assert_not_called()


if __name__ == "__main__":
    unittest.main()
