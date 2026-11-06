"""Кеш TensorRT-двигунів: manifest, режими збірки, блокування, перевірка двигуна."""
import json
import os
import threading
import time
from unittest import mock

import pytest

from modelhub_client_trt import engine_cache as ec

try:
    import tensorrt as trt
    import torch
    HAS_GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="потрібні CUDA і tensorrt")


# ---------- режим збірки ----------

@pytest.mark.parametrize("value,expected", [(None, "ifmissing"), ("", "ifmissing"), ("never", "never"),
                                            (" FORCE ", "force"), ("ifmissing", "ifmissing")])
def test_build_mode_from_env(value, expected):
    env = {} if value is None else {"MODELHUB_TRT_BUILD": value}
    assert ec.build_mode(env) == expected


def test_unknown_build_mode_is_an_error():
    with pytest.raises(ValueError):
        ec.build_mode({"MODELHUB_TRT_BUILD": "sometimes"})


# ---------- manifest ----------

def _engine(tmp_path, data=b"x" * 100):
    p = tmp_path / "m-GPU-trt10_13-bs1-fp16.engine"
    p.write_bytes(data)
    return str(p)


def test_manifest_matches_only_the_same_gpu_trt_and_file(tmp_path):
    e = _engine(tmp_path)
    ec.write_manifest(e, {"gpu_name": "NVIDIA RTX A6000", "trt_version": "10.13.0.35", "engine_bytes": 100})
    m = ec.read_manifest(e)
    assert ec.manifest_matches(m, e, "NVIDIA RTX A6000", "10.13.0.35")
    assert not ec.manifest_matches(m, e, "Quadro RTX 8000", "10.13.0.35")
    assert not ec.manifest_matches(m, e, "NVIDIA RTX A6000", "10.13.3.9")
    _engine(tmp_path, b"y" * 99)  # файл перезаписали, manifest лишився старий
    assert not ec.manifest_matches(m, e, "NVIDIA RTX A6000", "10.13.0.35")


def test_manifest_write_is_atomic_and_leaves_no_tmp(tmp_path):
    e = _engine(tmp_path)
    ec.write_manifest(e, {"gpu_name": "g"})
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(e), os.path.basename(e) + ".json"]
    assert json.load(open(e + ".json"))["schema"] == ec.MANIFEST_SCHEMA


def test_usable_fast_path_does_not_deserialize(tmp_path):
    e = _engine(tmp_path)
    ec.write_manifest(e, {"gpu_name": "g", "trt_version": "t", "engine_bytes": 100})
    with mock.patch.object(ec, "deserialize_check", side_effect=AssertionError("не мало викликатись")):
        ok, m, _ = ec.engine_is_usable(e, "g", "t")
    assert ok and m["gpu_name"] == "g"


def test_legacy_engine_without_manifest_is_checked_once_and_adopted(tmp_path):
    e = _engine(tmp_path)
    with mock.patch.object(ec, "deserialize_check", return_value=(True, 4096, "")) as check:
        ok, m, _ = ec.engine_is_usable(e, "g", "t", {"model_name": "m"})
        ok2, _, _ = ec.engine_is_usable(e, "g", "t")
    assert ok and ok2 and check.call_count == 1
    assert m["adopted"] is True and m["device_memory_bytes"] == 4096 and m["model_name"] == "m"
    assert ec.gpu_memory_estimate(m) == 100 + 4096


def test_engine_that_does_not_deserialize_is_rejected_with_reason(tmp_path):
    e = _engine(tmp_path)
    with mock.patch.object(ec, "deserialize_check", return_value=(False, None, "інша версія TensorRT")):
        ok, _, reason = ec.engine_is_usable(e, "g", "t")
    assert not ok and "TensorRT" in reason and not os.path.exists(e + ".json")


def test_manifest_from_another_trt_version_is_rejected_without_deserializing(tmp_path):
    e = _engine(tmp_path)
    ec.write_manifest(e, {"gpu_name": "g", "trt_version": "10.9", "engine_bytes": 100})
    with mock.patch.object(ec, "deserialize_check", side_effect=AssertionError):
        ok, _, reason = ec.engine_is_usable(e, "g", "10.13")
    assert not ok and "10.9" in reason


def test_missing_engine_is_not_usable(tmp_path):
    assert ec.engine_is_usable(str(tmp_path / "none.engine"), "g", "t")[0] is False


def test_ultralytics_metadata_header_is_stripped():
    meta = json.dumps({"stride": 32}).encode()
    blob = len(meta).to_bytes(4, "little") + meta + b"ENGINE"
    assert ec._engine_payload(blob) == b"ENGINE"
    assert ec._engine_payload(b"\x00\x01\x02\x03rawengine") == b"\x00\x01\x02\x03rawengine"


# ---------- блокування ----------

def test_lock_is_exclusive_and_released(tmp_path):
    p = str(tmp_path / "e.lock")
    a = ec.HeartbeatLock(p, wait_timeout_s=0.3, poll_s=0.05)
    b = ec.HeartbeatLock(p, wait_timeout_s=0.3, poll_s=0.05)
    assert a.acquire()
    with pytest.raises(TimeoutError):
        b.acquire()
    a.release()
    assert not os.path.exists(p)
    assert b.acquire()
    b.release()


def test_lock_left_by_a_killed_pod_is_taken_over(tmp_path):
    p = tmp_path / "e.lock"
    p.write_text("killed-pod:1")
    old = time.time() - 600
    os.utime(p, (old, old))
    lock = ec.HeartbeatLock(str(p), wait_timeout_s=1, stale_after_s=120, poll_s=0.05)
    assert lock.acquire()
    assert "killed-pod" not in p.read_text()
    lock.release()


def test_owner_heartbeat_keeps_a_long_build_alive(tmp_path):
    p = str(tmp_path / "e.lock")
    owner = ec.HeartbeatLock(p, stale_after_s=0.5, heartbeat_s=0.1)
    assert owner.acquire()
    time.sleep(1.0)  # довше за stale_after_s — але серцебиття оновлює mtime
    assert not ec.HeartbeatLock(p, stale_after_s=0.5).is_stale()
    owner.release()


def test_waiter_returns_when_another_process_finished_the_engine(tmp_path):
    p = str(tmp_path / "e.lock")
    owner = ec.HeartbeatLock(p)
    assert owner.acquire()
    done = threading.Event()
    threading.Timer(0.2, done.set).start()
    waiter = ec.HeartbeatLock(p, wait_timeout_s=5, poll_s=0.05)
    assert waiter.acquire(done=done.is_set) is False
    owner.release()


# ---------- справжній двигун на GPU ----------

def _tiny_engine(path):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = builder.create_network(flags)
    x = net.add_input("x", trt.float32, (1, 8))
    y = net.add_activation(x, trt.ActivationType.RELU)
    net.mark_output(y.get_output(0))
    cfg = builder.create_builder_config()
    blob = builder.build_serialized_network(net, cfg)
    with open(path, "wb") as f:
        f.write(bytes(blob))


@needs_gpu
def test_real_engine_deserializes_and_garbage_does_not(tmp_path):
    good = str(tmp_path / "good.engine")
    _tiny_engine(good)
    ok, mem, reason = ec.deserialize_check(good)
    assert ok and mem is not None and reason == ""
    bad = tmp_path / "bad.engine"
    bad.write_bytes(b"not an engine" * 100)
    ok, _, reason = ec.deserialize_check(str(bad))
    assert not ok and reason
