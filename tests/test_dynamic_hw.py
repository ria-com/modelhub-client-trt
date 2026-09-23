import os
import unittest
from unittest.mock import MagicMock, patch

from modelhub_client_trt.trt_converters.base import trt_dynamic_hw_enabled, trt_engine_suffix
from modelhub_client_trt.trt_converters.yolo import YoloConverter


class TestDynamicHwFlag(unittest.TestCase):
    def test_flag_is_off_by_default(self):
        self.assertFalse(trt_dynamic_hw_enabled({}))
        self.assertFalse(trt_dynamic_hw_enabled({"tensorrt": {"type": "yolo"}}))

    def test_flag_from_model_config(self):
        self.assertTrue(trt_dynamic_hw_enabled({"tensorrt": {"type": "yolo", "dynamic_hw": True}}))


class TestEngineSuffix(unittest.TestCase):
    def test_static_engine_suffix_is_unchanged(self):
        self.assertEqual(trt_engine_suffix({}, 1, True), "-bs1-fp16")
        self.assertEqual(trt_engine_suffix({"tensorrt": {"nms": True}}, 1, True), "-bs1-fp16-nms")

    def test_dynamic_hw_engine_gets_its_own_file_name(self):
        # a different input profile must never reuse a cached static engine for the same weights
        self.assertEqual(trt_engine_suffix({"tensorrt": {"dynamic_hw": True}}, 1, False), "-bs1-fp32-dynhw")


class TestLegacyYolov5DynamicExport(unittest.TestCase):
    def _convert(self, model_config):
        converter = YoloConverter.__new__(YoloConverter)
        with patch("modelhub_client_trt.trt_converters.yolo._ensure_yolov5_repo_cached", return_value="/repo"), \
                patch("modelhub_client_trt.trt_converters.yolo.shutil.copyfile"), \
                patch("modelhub_client_trt.trt_converters.yolo.os.path.exists", return_value=True), \
                patch("modelhub_client_trt.trt_converters.yolo.subprocess.run",
                      return_value=MagicMock(returncode=0, stdout="", stderr="")) as run, \
                patch("modelhub_client_trt.trt_converters.yolo.build_engine_from_onnx") as build:
            converter._convert_legacy_yolov5(
                "/models/weights.pt", "/models/out.engine", model_config, {},
                fp16_mode=True, max_batch_size=1, imgsz_h=640, imgsz_w=640,
            )
        return run.call_args[0][0], build.call_args.kwargs

    def test_static_export_by_default(self):
        cmd, build_kwargs = self._convert({})
        self.assertNotIn("--dynamic", cmd)
        self.assertIsNone(build_kwargs.get("max_hw"))

    def test_dynamic_hw_export_and_profile(self):
        cmd, build_kwargs = self._convert({"tensorrt": {"type": "yolo", "dynamic_hw": True}})
        self.assertIn("--dynamic", cmd)
        self.assertEqual(build_kwargs["max_hw"], (640, 640))


if __name__ == "__main__":
    unittest.main()
