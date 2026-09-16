import os
import sys
import tempfile
import unittest
from unittest import mock

from modelhub_client_trt.trt_converters import yolo


class TestEnsureYolov5RepoCached(unittest.TestCase):
    """torch.hub.load of ultralytics/yolov5 must never run in-process: it puts the
    repo's generic `models`/`utils` packages into sys.modules and breaks later
    YOLOv5 conversions (and consumers' own `models`/`utils` modules)."""

    def test_cold_cache_runs_hub_load_in_subprocess_only(self):
        with tempfile.TemporaryDirectory() as hub_dir:
            repo_dir = os.path.join(hub_dir, "ultralytics_yolov5_master")

            def fake_run(cmd, **kwargs):
                os.makedirs(repo_dir, exist_ok=True)
                open(os.path.join(repo_dir, "export.py"), "w").close()
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(yolo.torch.hub, "get_dir", return_value=hub_dir), \
                    mock.patch.object(yolo.torch.hub, "load", side_effect=AssertionError("in-process hub.load")), \
                    mock.patch.object(yolo.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(yolo._ensure_yolov5_repo_cached("w.pt"), repo_dir)

            cmd = run.call_args[0][0]
            self.assertEqual(cmd[0], sys.executable)
            self.assertIn("torch.hub.load", cmd[2])
            self.assertEqual(cmd[3], "w.pt")

    def test_warm_cache_does_not_spawn_anything(self):
        with tempfile.TemporaryDirectory() as hub_dir:
            repo_dir = os.path.join(hub_dir, "ultralytics_yolov5_master")
            os.makedirs(repo_dir)
            open(os.path.join(repo_dir, "export.py"), "w").close()
            with mock.patch.object(yolo.torch.hub, "get_dir", return_value=hub_dir), \
                    mock.patch.object(yolo.subprocess, "run") as run:
                self.assertEqual(yolo._ensure_yolov5_repo_cached("w.pt"), repo_dir)
            run.assert_not_called()

    def test_subprocess_failure_raises(self):
        with tempfile.TemporaryDirectory() as hub_dir:
            with mock.patch.object(yolo.torch.hub, "get_dir", return_value=hub_dir), \
                    mock.patch.object(yolo.subprocess, "run", return_value=mock.Mock(returncode=1, stderr="boom")):
                with self.assertRaises(RuntimeError):
                    yolo._ensure_yolov5_repo_cached("w.pt")


if __name__ == "__main__":
    unittest.main()
