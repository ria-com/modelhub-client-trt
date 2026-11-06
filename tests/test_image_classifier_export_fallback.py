import unittest

from modelhub_client_trt.trt_converters.image_classifier import _is_dynamo_exporter_error


class TestDynamoExporterFallback(unittest.TestCase):
    """torch >= 2.9 exports with dynamo by default; a broken onnx_ir/onnxscript combination
    must fall back to the legacy exporter instead of failing the whole engine build."""

    def test_onnx_ir_attribute_error(self):
        self.assertTrue(_is_dynamo_exporter_error(AttributeError("module 'onnx_ir' has no attribute 'schemas'")))

    def test_wrapped_in_chain(self):
        try:
            try:
                raise AttributeError("module 'onnx_ir' has no attribute 'schemas'")
            except AttributeError as inner:
                raise RuntimeError("export failed") from inner
        except RuntimeError as e:
            self.assertTrue(_is_dynamo_exporter_error(e))

    def test_unrelated_error_is_not_retried(self):
        self.assertFalse(_is_dynamo_exporter_error(RuntimeError("CUDA out of memory")))


if __name__ == "__main__":
    unittest.main()
