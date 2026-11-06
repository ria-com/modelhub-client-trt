# modelhub_client_trt/modelhub_client_trt.py
# ... (попередні імпорти без змін) ...
import os
import torch
import warnings
import time
from typing import Dict, List, Optional, Any
import glob
import shutil
import json
import socket

# --- Імпорти onnx, tensorrt, ModelHub etc. залишаються ---
try:
    import onnx
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    # ... (решта попереджень) ...

try:
    import tensorrt as trt
    _TENSORRT_AVAILABLE = True
    try:
        _TRT_VERSION_STR = f"{trt.getVersion()[0]}.{trt.getVersion()[1]}"
    except:
        try: _TRT_VERSION_STR = f"{trt.__version__.split('.')[0]}.{trt.__version__.split('.')[1]}"
        except: _TRT_VERSION_STR = "unknown"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    _TENSORRT_AVAILABLE = False
    _TRT_VERSION_STR = "unknown"
    warnings.warn("Бібліотека tensorrt не знайдена.")
    TRT_LOGGER = None
except Exception as e:
    _TENSORRT_AVAILABLE = False
    _TRT_VERSION_STR = "unknown"
    TRT_LOGGER = None
    warnings.warn(f"Не вдалося ініціалізувати TensorRT або визначити версію: {e}")

from modelhub_client import ModelHub

# --- Імпорт фабрики конвертерів ТА СЛОВНИКА КОНВЕРТЕРІВ ---
from .trt_converters.base import trt_engine_suffix
from .engine_cache import (TrtEngineMissing, HeartbeatLock, build_mode, deserialize_check,
                           engine_is_usable, gpu_compute_capability, manifest_path,
                           trt_full_version, write_manifest)
from .trt_converters import get_converter, BaseTrtConverter, TRT_CONVERTERS # <--- Додано TRT_CONVERTERS

from .trt_converters.yolo import _ULTRALYTICS_AVAILABLE

# --- Функції get_device_name, sanitize_for_filename (без змін) ---
def get_device_name() -> str:
    # ... (код без змін) ...
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception as e: warnings.warn(f"Не вдалося отримати назву GPU: {e}"); return "unknown_gpu"
    return "cpu"

def sanitize_for_filename(name: str) -> str:
    # ... (код без змін) ...
    name = name.replace(" ", "_")
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars: name = name.replace(char, '')
    name = name.replace("(", "").replace(")", "")
    name = name.replace('.', '_')
    return name
# --- Кінець незмінної частини ---


def auto_detect_converter_type(original_model_path: str, model_config: Dict[str, Any]) -> str:
    """
    Автоматичне визначення типу конвертера TensorRT для моделі, коли
    `tensorrt.type` не вказано явно у конфігурації моделі.

    Винесено в окрему (чисту, без мережі/GPU) функцію з `download_model_by_name_trt`,
    щоб гілки визначення архітектури можна було юніт-тестувати без завантаження
    моделі чи доступу до GPU.

    Правила (у порядку перевірки):
      1. Директорія з `config.json` -> 'text_classifier' (HuggingFace-стиль).
      2. `model_config['architecture']` містить 'yolo' -> 'yolo'.
      3. `model_config['architecture']` містить 'stn-ocr' -> 'crnn' (CTC-CRNN OCR
         sequence-модель — НЕ image_classifier).
      4. Файл закінчується на '.pt' -> 'image_classifier' (припускаємо TorchScript).
      5. Інакше -> 'image_classifier' (типовий фолбек за замовчуванням).
    """
    model_architecture = str(model_config.get("architecture", "")).lower()
    if os.path.isdir(original_model_path) and os.path.exists(os.path.join(original_model_path, 'config.json')):
        converter_type = 'text_classifier'
        print(f"Автоматично визначено тип конвертера як '{converter_type}' (папка з config.json).")
    elif "yolo" in model_architecture:
        # Конфіг моделі явно вказує архітектуру YOLO (напр. "yoloV5"/"yoloV8") у полі
        # "architecture" — надійніший сигнал, ніж просто розширення файлу, і не вимагає
        # спроби завантаження моделі для визначення типу.
        converter_type = 'yolo'
        print(f"Автоматично визначено тип конвертера як '{converter_type}' "
              f"(поле 'architecture'='{model_config.get('architecture')}' конфігу вказує на YOLO).")
    elif "stn-ocr" in model_architecture:
        # Конфіг моделі явно вказує архітектуру STN-OCR (CTC-CRNN sequence-модель)
        # — так само надійний сигнал, як і "yolo" вище.
        converter_type = 'crnn'
        print(f"Автоматично визначено тип конвертера як '{converter_type}' "
              f"(поле 'architecture'='{model_config.get('architecture')}' конфігу вказує на STN-OCR/CRNN).")
    elif original_model_path.endswith(".pt"):
        # .pt може бути YOLO або TorchScript image_classifier. Якщо архітектура з конфігу
        # не підказала YOLO/STN-OCR (гілки вище), безпечніше припустити image_classifier.
        converter_type = 'image_classifier'
        print(f"Автоматично визначено тип конвертера як '{converter_type}' (файл .pt, припускаємо TorchScript).")
    else:
        # Якщо нічого не підійшло, повертаємось до image_classifier за замовчуванням
        converter_type = "image_classifier"
        print(f"Не вдалося автоматично визначити тип конвертера. Використовується за замовчуванням: '{converter_type}'.")
    return converter_type

class ModelHubTrt(ModelHub):
    """
    Клас для управління моделями з підтримкою TensorRT оптимізації
    з використанням різних стратегій конвертації.
    """
    def __init__(self,
                 # ... (параметри init без змін) ...
                 models: Optional[Dict[str, Dict[str, str]]] = None,
                 model_config_urls: Optional[List[str]] = None,
                 local_storage: Optional[str] = None,
                 remote_storage: Optional[str] = None,
                 postfix: str = "./modelhub",
                 trt_build_wait_timeout: int = 3600,
                 trt_wait_interval: int = 5,
                 trt_onnx_opset: int = 19,
                 trt_builder_memory_limit_gb: int = 4,
                 trt_ultralytics_workspace_gb: int = 4
                 ) -> None:
        super().__init__(models=models, model_config_urls=model_config_urls, local_storage=local_storage,
                         remote_storage=remote_storage, postfix=postfix)
        # ... (решта ініціалізації без змін) ...
        self.trt_build_wait_timeout = trt_build_wait_timeout
        self.trt_wait_interval = trt_wait_interval
        self.trt_onnx_opset = trt_onnx_opset
        self.trt_builder_memory_limit = trt_builder_memory_limit_gb * (1024 ** 3)
        self.trt_ultralytics_workspace_gb = trt_ultralytics_workspace_gb
        if not _TENSORRT_AVAILABLE:
            warnings.warn("tensorrt не доступний. Функціонал TRT буде обмежено.")
            if not _ONNX_AVAILABLE:
                warnings.warn("onnx не доступний. Деякі типи конвертації TRT можуть не працювати.")
        print(f"ModelHubTrt ініціалізовано. Доступні конфігурації: {list(self.models.keys())}")

    def download_model_by_name_trt(self,
                                   model_name: str,
                                   path: Optional[str] = None,
                                   max_batch_size: int = 1,
                                   fp16_mode: bool = True) -> Dict[str, str]:
        """
        Завантажує модель та конвертує її в TensorRT .engine формат...
        """
        if not _TENSORRT_AVAILABLE:
             raise ImportError("Бібліотека tensorrt не встановлена або не ініціалізована.")

        print(f"Завантаження оригінальної моделі '{model_name}'...")
        model_info = super().download_model_by_name(model_name, path=path)
        original_model_path = model_info.get("path")
        if not original_model_path or not os.path.exists(original_model_path):
            raise FileNotFoundError(f"Оригінальний файл або директорію моделі не знайдено: {original_model_path}")
        print(f"Оригінальна модель завантажена: {original_model_path} (Is directory: {os.path.isdir(original_model_path)})")

        model_config = self.models.get(model_name)
        if not model_config:
            raise ValueError(f"Конфігурацію моделі не знайдено для '{model_name}' після завантаження.")

        # --- Debug: Виведемо конфіг (залишаємо для перевірки) ---
        # print(f"--- Debug: Конфігурація для '{model_name}': ---")
        # try: print(json.dumps(model_config, indent=2, ensure_ascii=False))
        # except Exception as e: print(f"(Не вдалося вивести JSON конфіг: {e})\n{model_config}")
        # print("--- End Debug ---")
        # ---

        # --- Визначення типу конвертера (ОНОВЛЕНА ЛОГІКА) ---
        converter_type = None
        tensorrt_config = model_config.get("tensorrt", {})
        explicit_type = tensorrt_config.get("type")

        if explicit_type:
            if explicit_type in TRT_CONVERTERS: # Перевіряємо, чи тип відомий
                converter_type = explicit_type
                print(f"Тип конвертера TensorRT '{converter_type}' взято з конфігурації моделі.")
            else:
                warnings.warn(f"Тип конвертера '{explicit_type}', вказаний у конфігурації для '{model_name}', "
                              f"не є відомим ({list(TRT_CONVERTERS.keys())}). Спроба автоматичного визначення.")

        # Якщо тип не було визначено з конфігу, спробуємо автоматично
        if converter_type is None:
            warnings.warn(f"Тип конвертера TensorRT не вказано або не розпізнано у конфігу для '{model_name}'. "
                          f"Спроба автоматичного визначення...")
            converter_type = auto_detect_converter_type(original_model_path, model_config)

        # Фінальна перевірка, чи отриманий тип існує
        if converter_type not in TRT_CONVERTERS:
             raise ValueError(f"Не вдалося визначити валідний тип конвертера TensorRT для моделі '{model_name}'. "
                              f"Визначено як '{converter_type}', доступні: {list(TRT_CONVERTERS.keys())}")

        print(f"Фінальний тип конвертера TensorRT: '{converter_type}'")
        # --- Кінець оновленої логіки визначення типу ---


        # --- Перевірка GPU та формування шляхів (без змін відносно попереднього кроку) ---
        gpu_name_raw = get_device_name()
        if gpu_name_raw == "cpu":
            raise RuntimeError("Оптимізація TensorRT потребує CUDA-сумісного GPU.")
        sanitized_gpu_name = sanitize_for_filename(gpu_name_raw)
        trt_version_sanitized = sanitize_for_filename(_TRT_VERSION_STR)

        if os.path.isdir(original_model_path):
            base_model_dir = original_model_path
            model_name_part = os.path.basename(original_model_path)
        else:
            base_model_dir = os.path.dirname(original_model_path)
            model_name_part, _ = os.path.splitext(os.path.basename(original_model_path))

        trt_target_dir = os.path.join(base_model_dir, sanitized_gpu_name)
        os.makedirs(trt_target_dir, exist_ok=True)
        # print(f"Директорія для файлів TensorRT: {trt_target_dir}") # Закоментуємо зайвий вивід

        engine_suffix = trt_engine_suffix(model_config, max_batch_size, fp16_mode)
        engine_file_name = (f"{model_name_part}-{sanitized_gpu_name}-trt{trt_version_sanitized}{engine_suffix}.engine")
        trt_engine_path = os.path.join(trt_target_dir, engine_file_name)
        lock_file_path = trt_engine_path + ".lock"
        onnx_file_path = os.path.join(trt_target_dir, f"{model_name_part}-temp{engine_suffix}.onnx")

        model_info["trt_engine_path"] = trt_engine_path
        model_info["trt_onnx_path"] = onnx_file_path
        model_info["gpu_name"] = gpu_name_raw
        model_info["tensorrt_converter_type"] = converter_type
        model_info["max_batch_size"] = max_batch_size
        model_info["fp16_mode"] = fp16_mode
        model_info['model_config'] = model_config
        # --- Кінець формування шляхів ---


        # --- Крок 4: готовий двигун, режим збірки, блокування, конвертація ---
        trt_version_full = trt_full_version()
        mode = build_mode()
        model_info["trt_version"] = trt_version_full
        model_info["build_mode"] = mode
        adopt_info = {
            "model_name": model_name, "converter_type": converter_type,
            "max_batch_size": max_batch_size, "fp16_mode": fp16_mode,
            "compute_capability": gpu_compute_capability(),
        }

        def usable():
            ok, manifest, reason = engine_is_usable(trt_engine_path, gpu_name_raw, trt_version_full, adopt_info)
            return ok, manifest, reason

        if mode != "force":
            ok, manifest, reason = usable()
            if ok:
                print(f"TensorRT двигун вже існує: {trt_engine_path}")
                self._cleanup_trt_dir(trt_target_dir, lock_file_path)
                model_info["trt_manifest"] = manifest
                return model_info
            if os.path.exists(trt_engine_path):
                warnings.warn(f"Двигун '{trt_engine_path}' не підходить ({reason}) — потрібна перебудова.")
            if mode == "never":
                raise TrtEngineMissing(
                    f"Немає готового TensorRT-двигуна для '{model_name}' на {gpu_name_raw} / TensorRT "
                    f"{trt_version_full}: {trt_engine_path} ({reason}). Збирає Job (MODELHUB_TRT_BUILD=ifmissing); "
                    f"у цьому процесі збірку заборонено (MODELHUB_TRT_BUILD=never).")

        print(f"Цільовий TensorRT двигун: {trt_engine_path}")
        lock = HeartbeatLock(lock_file_path, wait_timeout_s=self.trt_build_wait_timeout,
                             poll_s=self.trt_wait_interval)
        built_by_other = (lambda: usable()[0]) if mode != "force" else (lambda: False)
        try:
            if not lock.acquire(done=built_by_other):
                print("TensorRT двигун зібрав інший процес, поки ми чекали.")
                model_info["trt_manifest"] = usable()[1]
                self._cleanup_trt_dir(trt_target_dir, lock_file_path)
                return model_info
            if mode != "force" and usable()[0]:
                print(f"Двигун '{trt_engine_path}' з'явився до отримання блокування. Пропуск конвертації.")
            else:
                for stale in (trt_engine_path, manifest_path(trt_engine_path)):
                    if os.path.exists(stale):
                        os.remove(stale)
                print(f"\n--- Початок конвертації TensorRT (тип: {converter_type}) ---")
                print(f"GPU: {gpu_name_raw}, TensorRT {trt_version_full}")
                print(f"Оригінальна модель (шлях/папка): {original_model_path}")
                print(f"Параметри: max_batch_size={max_batch_size}, fp16={fp16_mode}")
                started = time.monotonic()
                try:
                    converter: BaseTrtConverter = get_converter(converter_type)
                    builder_config = {
                        'fp16_mode': fp16_mode,
                        'max_batch_size': max_batch_size,
                        'memory_limit': self.trt_builder_memory_limit,
                        'opset': self.trt_onnx_opset,
                        'ultralytics_workspace_gb': self.trt_ultralytics_workspace_gb,
                    }
                    converter.convert(
                        original_model_path=original_model_path,
                        engine_path=trt_engine_path,
                        onnx_path=onnx_file_path,
                        model_config=model_config,
                        builder_config=builder_config
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Помилка під час конвертації TRT (тип: {converter_type}): {e}") from e
                build_seconds = round(time.monotonic() - started, 1)
                ok, device_mem, reason = deserialize_check(trt_engine_path)
                if not ok:
                    raise RuntimeError(f"Зібраний двигун '{trt_engine_path}' не десеріалізується: {reason}")
                write_manifest(trt_engine_path, {
                    **adopt_info, "gpu_name": gpu_name_raw, "trt_version": trt_version_full,
                    "engine_bytes": os.path.getsize(trt_engine_path), "device_memory_bytes": device_mem,
                    "build_seconds": build_seconds, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "built_on": socket.gethostname(), "library_version": _library_version(),
                    "adopted": False,
                })
                print(f"--- Конвертація TensorRT (тип: {converter_type}) завершена за {build_seconds} с ---")
        finally:
            lock.release()
            if os.path.exists(trt_engine_path):
                self._cleanup_trt_dir(trt_target_dir, lock_file_path)
            else:
                for f in [onnx_file_path] + glob.glob(f"{os.path.splitext(onnx_file_path)[0]}.*.weight"):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        ok, manifest, reason = usable()
        if not ok:
            raise RuntimeError(f"Не вдалося створити або знайти TensorRT двигун '{trt_engine_path}': {reason}")
        model_info["trt_manifest"] = manifest
        print(f"Успішно отримано TensorRT двигун: {trt_engine_path}")
        return model_info

    @staticmethod
    def _cleanup_trt_dir(trt_target_dir: str, lock_file_path: str) -> None:
        """Прибирає проміжні файли збірки (ONNX, зовнішні ваги), лишає двигуни, manifest-и і чужі lock/tmp."""
        try:
            names = os.listdir(trt_target_dir)
        except OSError as e:
            warnings.warn(f"Не вдалося отримати список файлів у '{trt_target_dir}': {e}")
            return
        for filename in names:
            path = os.path.join(trt_target_dir, filename)
            if (filename.endswith((".engine", ".engine.json", ".lock")) or ".tmp" in filename
                    or path == lock_file_path or not os.path.isfile(path)):
                continue
            try:
                print(f"  - Видалення тимчасового файлу: {filename}")
                os.remove(path)
            except OSError as e:
                warnings.warn(f"Не вдалося видалити тимчасовий файл '{path}': {e}")


def _library_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


# --- Приклад використання __main__ (не змінюємо) ---
if __name__ == '__main__':
     # ... (код прикладу залишається без змін) ...
    # Перевірка доступності CUDA перед запуском прикладу
    if not torch.cuda.is_available():
        print("Пропуск прикладу: CUDA недоступний.")
    elif not _TENSORRT_AVAILABLE:
         print("Пропуск прикладу: TensorRT недоступний.")
    else:
        # Приклад 1: Модель Image Classifier (стандартна логіка)
        model_config_url_classifier = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
        model_name_classifier = "vehicle_registration_certificate_orientation_0_90_detector"

        # Приклад 2: Модель YOLO
        model_config_url_yolo = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/yolo_vrc/model-1.json"
        model_name_yolo = "yolo_vrc_detector"

        # Приклад 3: Модель Text Classifier
        model_config_url_text_classifier = "https://models.vsp.net.ua/config_model/speech-recognition:support-retrain/xlm-roberta-large-2024-03-24.json"
        model_name_text_classifier = "speech_support"


        local_storage_path = os.environ.get('LOCAL_STORAGE', "./downloaded_models_trt_refactored")
        os.makedirs(local_storage_path, exist_ok=True)
        print(f"Локальне сховище: {local_storage_path}")
        print(f"Версія TensorRT: {_TRT_VERSION_STR}")
        print(f"GPU: {get_device_name()}")

        try:
            modelhub_trt = ModelHubTrt(model_config_urls=[
                                            model_config_url_classifier,
                                            model_config_url_yolo,
                                            model_config_url_text_classifier # Додано сюди
                                        ],
                                       local_storage=local_storage_path,
                                       trt_build_wait_timeout=600,
                                       trt_onnx_opset=19,
                                       trt_builder_memory_limit_gb=8, # Збільшено для трансформерів
                                       trt_ultralytics_workspace_gb=4)

            # --- Тест 1: Image Classifier ---
            print(f"\n=== ТЕСТ 1: Запуск для Image Classifier ({model_name_classifier}) ===")
            optimized_model_info_cls = modelhub_trt.download_model_by_name_trt(
                model_name_classifier,
                max_batch_size=1,
                fp16_mode=True
            )
            print("\n--- Результат для Image Classifier ---")
            print(f"Оригінальний шлях: {optimized_model_info_cls.get('path')}")
            print(f"GPU для оптимізації: {optimized_model_info_cls.get('gpu_name')}")
            print(f"Шлях до TRT двигуна: {optimized_model_info_cls.get('trt_engine_path')}")
            print(f"Тип конвертера: {optimized_model_info_cls.get('tensorrt_converter_type')}")
            print(f"Max Batch Size: {optimized_model_info_cls.get('max_batch_size')}")
            print(f"FP16 Mode: {optimized_model_info_cls.get('fp16_mode')}")

            # --- Тест 2: YOLO ---
            if _ULTRALYTICS_AVAILABLE:
                print(f"\n=== ТЕСТ 2: Запуск для YOLO ({model_name_yolo}) ===")
                optimized_model_info_yolo = modelhub_trt.download_model_by_name_trt(
                    model_name_yolo,
                    max_batch_size=1,
                    fp16_mode=True
                )
                print("\n--- Результат для YOLO ---")
                print(f"Оригінальний шлях: {optimized_model_info_yolo.get('path')}")
                print(f"GPU для оптимізації: {optimized_model_info_yolo.get('gpu_name')}")
                print(f"Шлях до TRT двигуна: {optimized_model_info_yolo.get('trt_engine_path')}")
                print(f"Тип конвертера: {optimized_model_info_yolo.get('tensorrt_converter_type')}")
                print(f"Max Batch Size: {optimized_model_info_yolo.get('max_batch_size')}")
                print(f"FP16 Mode: {optimized_model_info_yolo.get('fp16_mode')}")
            else:
                print("\n=== ТЕСТ 2: Пропущено (YOLO) - бібліотека ultralytics не знайдена ===")

            # --- Тест 3: Text Classifier ---
            print(f"\n=== ТЕСТ 3: Запуск для Text Classifier ({model_name_text_classifier}) ===")
            optimized_model_info_text = modelhub_trt.download_model_by_name_trt(
                model_name_text_classifier,
                max_batch_size=2, # Спробуємо з батчем 2
                fp16_mode=True    # і FP16
            )
            print("\n--- Результат для Text Classifier ---")
            print(f"Оригінальний шлях: {optimized_model_info_text.get('path')}")
            print(f"GPU для оптимізації: {optimized_model_info_text.get('gpu_name')}")
            print(f"Шлях до TRT двигуна: {optimized_model_info_text.get('trt_engine_path')}")
            print(f"Тип конвертера: {optimized_model_info_text.get('tensorrt_converter_type')}")
            print(f"Max Batch Size: {optimized_model_info_text.get('max_batch_size')}")
            print(f"FP16 Mode: {optimized_model_info_text.get('fp16_mode')}")
            # Виведемо конфіг, переданий з download_model_by_name_trt
            # print("\nПовна конфігурація моделі (з model_info):")
            # model_cfg_print = optimized_model_info_text.get('model_config', {})
            # try: print(json.dumps(model_cfg_print, indent=2, ensure_ascii=False))
            # except: print(model_cfg_print)


        except ImportError as e: print(f"\nПОМИЛКА: Помилка імпорту: {e}.")
        except FileNotFoundError as e: print(f"\nПОМИЛКА: Файл не знайдено - {e}")
        except ValueError as e: print(f"\nПОМИЛКА: Невірні дані - {e}")
        except TimeoutError as e: print(f"\nПОМИЛКА: Таймаут - {e}")
        except RuntimeError as e: print(f"\nПОМИЛКА: Помилка виконання - {e}"); import traceback; traceback.print_exc()
        except Exception as e: print(f"\nПОМИЛКА: Неочікувана помилка - {e}"); import traceback; traceback.print_exc()