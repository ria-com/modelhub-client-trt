import unittest

from modelhub_client_trt.modelhub_client_trt import auto_detect_converter_type
from modelhub_client_trt.trt_converters import TRT_CONVERTERS
from modelhub_client_trt.trt_converters.base import trt_export_nms_enabled


class TestAutoDetectConverterType(unittest.TestCase):
    """
    Unit tests for `auto_detect_converter_type` (the auto-detection branch used
    by `ModelHubTrt.download_model_by_name_trt` when a model's config does not
    explicitly set `tensorrt.type`). No network/GPU access required — the
    function only inspects `model_config`/the (non-existent, for these tests)
    `original_model_path` string.
    """

    def test_stn_ocr_architecture_routes_to_crnn(self):
        model_config = {"architecture": "STN-OCR"}
        converter_type = auto_detect_converter_type("does_not_matter.ckpt", model_config)
        self.assertEqual(converter_type, "crnn")

    def test_stn_ocr_architecture_is_case_insensitive(self):
        for architecture in ("stn-ocr", "Stn-Ocr", "STN-ocr-v2", "sTn-OcR"):
            with self.subTest(architecture=architecture):
                model_config = {"architecture": architecture}
                converter_type = auto_detect_converter_type("does_not_matter.ckpt", model_config)
                self.assertEqual(converter_type, "crnn")

    def test_crnn_converter_type_is_registered(self):
        self.assertIn("crnn", TRT_CONVERTERS)

    def test_yolo_architecture_still_routes_to_yolo(self):
        model_config = {"architecture": "yoloV8"}
        converter_type = auto_detect_converter_type("does_not_matter.pt", model_config)
        self.assertEqual(converter_type, "yolo")

    def test_pt_file_without_known_architecture_routes_to_image_classifier(self):
        model_config = {"architecture": "some-unknown-cnn"}
        converter_type = auto_detect_converter_type("does_not_matter.pt", model_config)
        self.assertEqual(converter_type, "image_classifier")

    def test_unknown_everything_falls_back_to_image_classifier(self):
        model_config = {}
        converter_type = auto_detect_converter_type("does_not_matter.bin", model_config)
        self.assertEqual(converter_type, "image_classifier")


class TestTrtExportNmsEnabled(unittest.TestCase):
    def test_enabled_only_via_tensorrt_block(self):
        self.assertTrue(trt_export_nms_enabled({"tensorrt": {"type": "yolo", "nms": True}}))
        self.assertFalse(trt_export_nms_enabled({"tensorrt": {"type": "yolo"}}))
        self.assertFalse(trt_export_nms_enabled({}))
        self.assertFalse(trt_export_nms_enabled({"tensorrt": None}))


if __name__ == "__main__":
    unittest.main()
