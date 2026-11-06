"""download_model_by_name_trt: режими збірки, старі двигуни без manifest, конкурентні збирачі.

Конвертер підмінено фейковим, який пише справжній крихітний TensorRT-двигун, — тож перевірка
десеріалізацією і manifest працюють по-справжньому, а мережі й моделей не треба.
"""
import os
import threading
from unittest import mock

import pytest

from modelhub_client_trt import modelhub_client_trt as mod
from modelhub_client_trt.engine_cache import TrtEngineMissing, read_manifest

from tests.test_engine_cache import HAS_GPU, _tiny_engine

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="потрібні CUDA і tensorrt")


class FakeConverter:
    calls = 0
    lock = threading.Lock()
    delay = 0.0

    def convert(self, original_model_path, engine_path, onnx_path, model_config, builder_config):
        with FakeConverter.lock:
            FakeConverter.calls += 1
        open(onnx_path, "w").write("onnx")           # проміжний файл, має бути прибраний
        if FakeConverter.delay:
            threading.Event().wait(FakeConverter.delay)
        tmp = engine_path + ".tmp"
        _tiny_engine(tmp)
        os.replace(tmp, engine_path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    FakeConverter.calls, FakeConverter.delay = 0, 0.0
    model_dir = tmp_path / "models" / "m"
    model_dir.mkdir(parents=True)
    weights = model_dir / "m.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(mod.ModelHub, "download_model_by_name",
                        lambda self, name, path=None: {"path": str(weights)})
    monkeypatch.setattr(mod, "get_converter", lambda t: FakeConverter())
    monkeypatch.delenv("MODELHUB_TRT_BUILD", raising=False)
    h = mod.ModelHubTrt(models={"m": {"tensorrt": {"type": "image_classifier"}}},
                        local_storage=str(tmp_path / "models"), remote_storage="none", trt_wait_interval=0.05)
    return h


def test_first_call_builds_and_writes_manifest(hub):
    info = hub.download_model_by_name_trt("m")
    m = read_manifest(info["trt_engine_path"])
    assert FakeConverter.calls == 1
    assert m["gpu_name"] == info["gpu_name"] and m["trt_version"] == info["trt_version"]
    assert m["adopted"] is False and m["device_memory_bytes"] is not None and m["library_version"] == "1.4.0"
    assert sorted(os.listdir(os.path.dirname(info["trt_engine_path"]))) == sorted(
        [os.path.basename(info["trt_engine_path"]), os.path.basename(info["trt_engine_path"]) + ".json"])


def test_second_call_reuses_without_building(hub):
    hub.download_model_by_name_trt("m")
    info = hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 1 and info["trt_manifest"]["adopted"] is False


def test_never_mode_refuses_to_build(hub, monkeypatch):
    monkeypatch.setenv("MODELHUB_TRT_BUILD", "never")
    with pytest.raises(TrtEngineMissing, match="MODELHUB_TRT_BUILD=never"):
        hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 0


def test_never_mode_uses_an_engine_built_by_the_job(hub, monkeypatch):
    hub.download_model_by_name_trt("m")              # це робить Job
    monkeypatch.setenv("MODELHUB_TRT_BUILD", "never")  # а це — робочий под
    assert hub.download_model_by_name_trt("m")["trt_manifest"]["gpu_name"]
    assert FakeConverter.calls == 1


def test_force_mode_rebuilds(hub, monkeypatch):
    hub.download_model_by_name_trt("m")
    monkeypatch.setenv("MODELHUB_TRT_BUILD", "force")
    hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 2


def test_engine_from_older_library_is_adopted_not_rebuilt(hub):
    info = hub.download_model_by_name_trt("m")
    os.remove(info["trt_engine_path"] + ".json")     # як двигун, зібраний 1.3.0
    info = hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 1 and info["trt_manifest"]["adopted"] is True


def test_broken_engine_is_rebuilt(hub):
    info = hub.download_model_by_name_trt("m")
    os.remove(info["trt_engine_path"] + ".json")
    open(info["trt_engine_path"], "wb").write(b"truncated")
    hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 2


def test_engine_for_another_trt_version_is_rebuilt(hub):
    info = hub.download_model_by_name_trt("m")
    with mock.patch.object(mod, "trt_full_version", return_value="99.0.0"):
        info2 = hub.download_model_by_name_trt("m")
    assert FakeConverter.calls == 2 and info2["trt_manifest"]["trt_version"] == "99.0.0"


def test_concurrent_callers_build_once(hub):
    FakeConverter.delay = 0.5
    results, errors = [], []

    def call():
        try:
            results.append(hub.download_model_by_name_trt("m"))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 4 and FakeConverter.calls == 1


def test_cleanup_keeps_json_files_next_to_the_engine(hub):
    info = hub.download_model_by_name_trt("m")
    side = info["trt_engine_path"] + ".io.json"       # кеш входів/виходів, який пише проєкт
    open(side, "w").write("{}")
    os.remove(info["trt_engine_path"] + ".json")
    hub.download_model_by_name_trt("m")              # шлях з очищенням каталогу
    assert os.path.exists(side)
