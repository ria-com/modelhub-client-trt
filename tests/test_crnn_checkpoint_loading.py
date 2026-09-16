import os
import sys
import tempfile
import types
import unittest

import torch

from modelhub_client_trt.trt_converters.crnn import _load_checkpoint


class TestCrnnCheckpointLoading(unittest.TestCase):
    """Lightning checkpoints may pickle objects from training code (e.g. a label
    converter in `hyper_parameters`). Loading must still work where that code is
    not importable, since only `state_dict` and plain hyper-parameters are used."""

    def test_missing_training_module_is_replaced_with_stub(self):
        module_name = "_training_only_module_for_test"
        mod = types.ModuleType(module_name)

        class LabelConverter:
            def __init__(self, alphabet):
                self.alphabet = alphabet

        LabelConverter.__module__ = module_name
        LabelConverter.__qualname__ = "LabelConverter"
        mod.LabelConverter = LabelConverter
        sys.modules[module_name] = mod

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.ckpt")
            try:
                torch.save({
                    "state_dict": {"linear.weight": torch.ones(2, 3)},
                    "hyper_parameters": {"letters": ["0", "1"], "hidden_size": 32,
                                         "label_converter": LabelConverter("01")},
                }, path)
            finally:
                del sys.modules[module_name]

            ckpt = _load_checkpoint(path)

        self.assertTrue(torch.equal(ckpt["state_dict"]["linear.weight"], torch.ones(2, 3)))
        self.assertEqual(ckpt["hyper_parameters"]["letters"], ["0", "1"])
        self.assertEqual(ckpt["hyper_parameters"]["hidden_size"], 32)
        self.assertEqual(type(ckpt["hyper_parameters"]["label_converter"]).__name__, "LabelConverter")

    def test_regular_checkpoint_loads_without_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.ckpt")
            torch.save({"state_dict": {"w": torch.zeros(1)}, "hyper_parameters": {"letters": ["a"]}}, path)
            ckpt = _load_checkpoint(path)
        self.assertEqual(ckpt["hyper_parameters"], {"letters": ["a"]})


if __name__ == "__main__":
    unittest.main()
