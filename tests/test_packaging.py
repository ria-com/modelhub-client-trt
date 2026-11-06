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
