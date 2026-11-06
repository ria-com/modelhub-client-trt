"""Пакет не тягне torch/tensorrt: їх дає базовий образ, і вимога `torch` ламала NGC-образи."""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def requirements():
    import runpy, setuptools
    seen = {}
    orig = setuptools.setup
    setuptools.setup = lambda **kw: seen.update(kw)
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        runpy.run_path(os.path.join(ROOT, "setup.py"))
    finally:
        setuptools.setup = orig
        os.chdir(cwd)
    return seen["install_requires"]


def test_requirements_are_one_per_line():
    reqs = requirements()
    assert all("\n" not in r for r in reqs) and len(reqs) >= 1


def test_torch_and_tensorrt_are_not_required():
    names = {re.split(r"[ <>=@\[]", r, maxsplit=1)[0].strip().lower() for r in requirements()}
    assert not names & {"torch", "tensorrt", "torchvision", "onnxruntime-gpu", "torch2trt"}
    assert "modelhub_client" in names


def test_onnxslim_is_required_for_ultralytics_export():
    names = {re.split(r"[ <>=@\[]", r, maxsplit=1)[0].strip().lower() for r in requirements()}
    assert "onnxslim" in names


def test_yolo_converter_disables_ultralytics_autoinstall(monkeypatch):
    import importlib, sys
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    sys.modules.pop("modelhub_client_trt.trt_converters.yolo", None)
    importlib.import_module("modelhub_client_trt.trt_converters.yolo")
    import os
    assert os.environ["YOLO_AUTOINSTALL"] == "False"


def test_yolo_export_runs_from_a_private_copy(tmp_path, monkeypatch):
    """Дві паралельні збірки однієї моделі не пишуть ONNX у спільний каталог .pt."""
    from modelhub_client_trt.trt_converters import yolo
    shared = tmp_path / "model"; shared.mkdir()
    pt = shared / "m.pt"; pt.write_bytes(b"weights")
    seen = []

    def fake(self, local, engine_path, onnx_path, model_config, builder_config):
        seen.append(local)
        assert os.path.dirname(local) != str(shared)
        assert open(local, "rb").read() == b"weights"
        open(engine_path, "wb").write(b"engine")

    monkeypatch.setattr(yolo.YoloConverter, "_convert_from", fake)
    for gpu in ("A6000", "RTX8000"):
        d = tmp_path / gpu; d.mkdir()
        yolo.YoloConverter().convert(str(pt), str(d / "m.engine"), None, {}, {})
        assert sorted(os.listdir(d)) == ["m.engine"]          # приватну копію прибрано
    assert seen[0] != seen[1] and os.listdir(shared) == ["m.pt"]
