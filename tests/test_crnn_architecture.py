"""CRNN converter: architecture and input size come from the checkpoint/config,
not from constants (VIN OCR: Linear(2816, 512), 64x2048; cadastral OCR:
Linear(2816, 131), hidden 35, 53x1275, `conv_nn`/`recurrent_layer*.rnn` names)."""
import unittest

import torch
import torch.nn as nn

from modelhub_client_trt.modelhub_client_trt import auto_detect_converter_type
from modelhub_client_trt.trt_converters.crnn import (_CrnnOcrNet, crnn_arch_from_state_dict,
                                                      crnn_input_hw, crnn_model_from_checkpoint)


def _reference_state_dict(linear_in, linear_out, hidden, letters_max, names="vin"):
    """state_dict of the training-code module with the given sizes (random weights)."""
    torch.manual_seed(0)
    net = _CrnnOcrNet(letters_max=letters_max, hidden_size=hidden, linear_in=linear_in, linear_out=linear_out)
    sd = net.state_dict()
    if names == "cadastral":
        renamed = {}
        for k, v in sd.items():
            k = k.replace("backbone_app.", "conv_nn.", 1)
            k = k.replace("gru1.gru.", "recurrent_layer1.rnn.", 1).replace("gru2.gru.", "recurrent_layer2.rnn.", 1)
            renamed[k] = v
        sd = renamed
    return sd


class TestArchFromStateDict(unittest.TestCase):
    def test_vin_sizes(self):
        sd = _reference_state_dict(2816, 512, 32, 34)
        arch = crnn_arch_from_state_dict(sd)
        self.assertEqual((arch["linear_in"], arch["linear_out"], arch["hidden_size"], arch["letters_max"]),
                         (2816, 512, 32, 34))
        self.assertTrue(arch["bidirectional"])

    def test_cadastral_names_and_sizes(self):
        sd = _reference_state_dict(2816, 131, 35, 11, names="cadastral")
        arch = crnn_arch_from_state_dict(sd)
        self.assertEqual((arch["linear_in"], arch["linear_out"], arch["hidden_size"], arch["letters_max"]),
                         (2816, 131, 35, 11))
        self.assertTrue(all(k.startswith(("backbone_app.", "linear1.", "gru1.", "gru2.", "linear2."))
                            for k in arch["state_dict"]))

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            crnn_arch_from_state_dict({"foo.weight": torch.zeros(1)})


class TestInputHw(unittest.TestCase):
    def test_image_size_keys_win(self):
        self.assertEqual(crnn_input_hw({"image_size_h": 50, "image_size_w": 200, "height": 64, "width": 2048}, {}),
                         (50, 200))

    def test_height_width_keys(self):
        self.assertEqual(crnn_input_hw({"height": 64, "width": 2048}, {}), (64, 2048))

    def test_hyper_parameters_fallback(self):
        self.assertEqual(crnn_input_hw({}, {"height": 53, "width": 1275}), (53, 1275))

    def test_legacy_default(self):
        self.assertEqual(crnn_input_hw({}, {}), (128, 384))


class TestModelFromCheckpoint(unittest.TestCase):
    def test_vin_checkpoint_matches_reference_outputs(self):
        sd = _reference_state_dict(2816, 512, 32, 34)
        ref = _CrnnOcrNet(letters_max=34, hidden_size=32, linear_in=2816, linear_out=512)
        ref.load_state_dict(sd)
        ref.eval()
        model, (h, w) = crnn_model_from_checkpoint({"state_dict": sd, "hyper_parameters": {}},
                                                   {"height": 64, "width": 320})
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), ref(x)))
        self.assertEqual(tuple(model(x).shape), (10, 1, 34))

    def test_cadastral_checkpoint_loads(self):
        sd = _reference_state_dict(2816, 131, 35, 11, names="cadastral")
        model, hw = crnn_model_from_checkpoint({"state_dict": sd, "hyper_parameters": {"height": 53, "width": 1275}}, {})
        self.assertEqual(hw, (53, 1275))
        self.assertEqual(model.linear1.out_features, 131)

    def test_input_size_mismatch_is_reported(self):
        sd = _reference_state_dict(2816, 512, 32, 34)
        with self.assertRaises(ValueError) as cm:
            crnn_model_from_checkpoint({"state_dict": sd, "hyper_parameters": {}}, {"image_size_h": 128, "image_size_w": 384})
        self.assertIn("2816", str(cm.exception))


class TestCkptRouting(unittest.TestCase):
    def test_lightning_ocr_checkpoint_routes_to_crnn(self):
        self.assertEqual(auto_detect_converter_type("vin_ocr.ckpt", {"letters": ["0", "1"]}), "crnn")

    def test_ckpt_without_letters_keeps_default(self):
        self.assertEqual(auto_detect_converter_type("model.ckpt", {}), "image_classifier")


if __name__ == "__main__":
    unittest.main()
