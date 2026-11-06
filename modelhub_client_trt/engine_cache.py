# modelhub_client_trt/engine_cache.py
"""
Кеш TensorRT-двигунів: чи можна взяти готовий двигун, хто його будує і що про нього відомо.

Двигун TensorRT залежить від версії TensorRT і моделі GPU, на якій його зібрано, але не від
драйвера. Тому поруч із кожним двигуном лежить manifest `<engine>.json` (GPU, compute capability,
повна версія TensorRT, скільки пам'яті двигун займає), і двигун береться лише тоді, коли manifest
збігається з поточним GPU і TensorRT. Двигун без manifest (зібраний старішою версією бібліотеки)
перевіряється десеріалізацією один раз, і йому дописується manifest.

Режим збірки — змінна оточення MODELHUB_TRT_BUILD:
  ifmissing (за замовчуванням) — взяти готовий або зібрати;
  never  — лише взяти готовий, інакше TrtEngineMissing (робочі поди: збирає окрема Job);
  force  — перебудувати, навіть якщо готовий є.
"""
import json
import os
import socket
import threading
import time
import warnings
from typing import Any, Dict, Optional, Tuple

BUILD_MODES = ("ifmissing", "never", "force")
MANIFEST_SCHEMA = 1


class TrtEngineMissing(RuntimeError):
    """Готового двигуна для цього GPU і TensorRT немає, а збирати заборонено (MODELHUB_TRT_BUILD=never)."""


def build_mode(env=os.environ) -> str:
    mode = (env.get("MODELHUB_TRT_BUILD") or "ifmissing").strip().lower()
    if mode not in BUILD_MODES:
        raise ValueError(f"MODELHUB_TRT_BUILD={mode!r}, очікується одне з {BUILD_MODES}")
    return mode


def trt_full_version() -> str:
    try:
        import tensorrt as trt
        return str(trt.__version__)
    except Exception:
        return "unknown"


def gpu_compute_capability() -> str:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"{major}.{minor}"
    except Exception:
        return "unknown"


def manifest_path(engine_path: str) -> str:
    return engine_path + ".json"


def read_manifest(engine_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(manifest_path(engine_path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_manifest(engine_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    data = {"schema": MANIFEST_SCHEMA, **data}
    tmp = f"{manifest_path(engine_path)}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path(engine_path))
    return data


def manifest_matches(manifest: Optional[Dict[str, Any]], engine_path: str,
                     gpu_name: str, trt_version: str) -> bool:
    """Manifest описує саме цей файл (розмір) і саме цей GPU та TensorRT."""
    if not manifest:
        return False
    try:
        size = os.path.getsize(engine_path)
    except OSError:
        return False
    return (manifest.get("gpu_name") == gpu_name
            and manifest.get("trt_version") == trt_version
            and manifest.get("engine_bytes") == size)


def _engine_payload(blob: bytes) -> bytes:
    """Двигун, експортований ultralytics, має спереду JSON-метадані: 4 байти довжини + JSON."""
    if len(blob) > 4:
        n = int.from_bytes(blob[:4], "little")
        if 0 < n < min(len(blob) - 4, 1 << 20) and blob[4:5] == b"{":
            try:
                json.loads(blob[4:4 + n].decode("utf-8"))
                return blob[4 + n:]
            except (UnicodeDecodeError, ValueError):
                pass
    return blob


# TensorRT десеріалізує двигун, зібраний на ІНШІЙ моделі GPU (того ж покоління), і лише попереджає —
# «likely to affect performance or even cause errors or deadlock». Такий двигун не годиться.
_FOREIGN_DEVICE_MARK = "across different models of devices"


def _capturing_logger(trt):
    class _Capture(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.WARNING:
                self.messages.append(str(msg))
    return _Capture()


def deserialize_check(engine_path: str) -> Tuple[bool, Optional[int], str]:
    """
    (чи придатний, device memory у байтах, причина). Потрібен GPU і tensorrt.
    Непридатний і той, що десеріалізувався, але TensorRT каже, що його зібрано на іншій моделі GPU.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        return False, None, f"tensorrt недоступний: {e}"
    try:
        with open(engine_path, "rb") as f:
            payload = _engine_payload(f.read())
        logger = _capturing_logger(trt)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            return False, None, "deserialize_cuda_engine повернув None (інша версія TensorRT або GPU)"
        mem = getattr(engine, "device_memory_size_v2", None)
        if mem is None:
            mem = getattr(engine, "device_memory_size", None)
        del engine
        if any(_FOREIGN_DEVICE_MARK in m for m in logger.messages):
            return False, None, "зібраний на іншій моделі GPU (TensorRT: plan file across different models of devices)"
        return True, (int(mem) if mem is not None else None), ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def engine_is_usable(engine_path: str, gpu_name: str, trt_version: str,
                     adopt_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Чи можна віддати двигун. Швидкий шлях — manifest збігається. Інакше десеріалізація;
    якщо вона вдалась, двигун «усиновлюється»: дописується manifest (adopted=True).
    """
    if not os.path.exists(engine_path):
        return False, None, "немає файлу"
    manifest = read_manifest(engine_path)
    if manifest_matches(manifest, engine_path, gpu_name, trt_version) and manifest.get("device_checked"):
        return True, manifest, ""
    if manifest and (manifest.get("gpu_name") != gpu_name or manifest.get("trt_version") != trt_version):
        return False, manifest, (f"зібраний для {manifest.get('gpu_name')} / TensorRT {manifest.get('trt_version')}, "
                                 f"а тут {gpu_name} / TensorRT {trt_version}")
    ok, device_mem, reason = deserialize_check(engine_path)
    if not ok:
        return False, manifest, reason
    data = dict(manifest or {}) if manifest_matches(manifest, engine_path, gpu_name, trt_version) else {}
    data.update(adopt_info or {})
    data.update(gpu_name=gpu_name, trt_version=trt_version, engine_bytes=os.path.getsize(engine_path),
                device_memory_bytes=device_mem, device_checked=True)
    data.setdefault("adopted", True)
    try:
        manifest = write_manifest(engine_path, data)
    except OSError as e:
        warnings.warn(f"Не вдалося записати manifest для '{engine_path}': {e}")
        manifest = data
    return True, manifest, ""


def gpu_memory_estimate(manifest: Dict[str, Any]) -> Optional[int]:
    """Ваги (≈ розмір файлу) + робоча пам'ять двигуна; без CUDA-контексту процесу."""
    if manifest.get("engine_bytes") is None or manifest.get("device_memory_bytes") is None:
        return None
    return int(manifest["engine_bytes"]) + int(manifest["device_memory_bytes"])


class HeartbeatLock:
    """
    Блокування збірки файлом на спільному томі (PVC/NFS), з серцебиттям.

    Власник раз на `heartbeat_s` оновлює mtime файлу. Чужий lock вважається покинутим, коли
    mtime старший за `stale_after_s` (под убили посеред збірки) — тоді його забирають.
    Поки lock живий, чекають до `wait_timeout_s`, а не падають через фіксовані 5 хвилин.
    """

    def __init__(self, path: str, wait_timeout_s: float = 3600, stale_after_s: float = 120,
                 heartbeat_s: float = 15, poll_s: float = 5):
        self.path = path
        self.wait_timeout_s = wait_timeout_s
        self.stale_after_s = stale_after_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.acquired = False

    def _owner(self) -> str:
        return f"{socket.gethostname()}:{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}"

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(self._owner())
        return True

    def is_stale(self, now: Optional[float] = None) -> bool:
        try:
            age = (now if now is not None else time.time()) - os.path.getmtime(self.path)
        except OSError:
            return False
        return age > self.stale_after_s

    def acquire(self, done=lambda: False) -> bool:
        """
        True — lock наш, будуємо. False — поки чекали, `done()` став True (двигун зібрав хтось інший).
        TimeoutError — живий lock тримали довше за wait_timeout_s.
        """
        deadline = time.monotonic() + self.wait_timeout_s
        while True:
            if done():
                return False
            if self._try_create():
                self.acquired = True
                self._thread = threading.Thread(target=self._beat, daemon=True)
                self._thread.start()
                return True
            if self.is_stale():
                warnings.warn(f"Lock '{self.path}' без серцебиття понад {self.stale_after_s:.0f} с — забираю.")
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Lock '{self.path}' тримають довше за {self.wait_timeout_s:.0f} с.")
            time.sleep(self.poll_s)

    def _beat(self):
        while not self._stop.wait(self.heartbeat_s):
            try:
                os.utime(self.path, None)
            except OSError:
                return

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.heartbeat_s + 1)
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
