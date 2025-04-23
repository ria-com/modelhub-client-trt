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
                 trt_build_wait_timeout: int = 300,
                 trt_wait_interval: int = 5,
                 trt_onnx_opset: int = 17,
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
            if os.path.isdir(original_model_path) and os.path.exists(os.path.join(original_model_path, 'config.json')):
                 converter_type = 'text_classifier'
                 print(f"Автоматично визначено тип конвертера як '{converter_type}' (папка з config.json).")
            elif original_model_path.endswith(".pt"):
                 # .pt може бути YOLO або TorchScript image_classifier. Безпечніше припустити image_classifier.
                 converter_type = 'image_classifier'
                 print(f"Автоматично визначено тип конвертера як '{converter_type}' (файл .pt, припускаємо TorchScript).")
                 # Якщо ultralytics доступний, можна було б додати перевірку, але це ускладнить логіку
                 # if _ULTRALYTICS_AVAILABLE:
                 #     try:
                 #         # Спроба завантажити як YOLO без повного створення об'єкта
                 #         from ultralytics.nn.tasks import attempt_load_one_weight
                 #         attempt_load_one_weight(original_model_path)
                 #         converter_type = 'yolo'
                 #         print(f"Автоматично визначено тип конвертера як '{converter_type}' (файл .pt схожий на YOLO).")
                 #     except:
                 #         print(f"Файл .pt не схожий на YOLO, залишаємо '{converter_type}'.")
            else:
                 # Якщо нічого не підійшло, повертаємось до image_classifier за замовчуванням
                 converter_type = "image_classifier"
                 print(f"Не вдалося автоматично визначити тип конвертера. Використовується за замовчуванням: '{converter_type}'.")

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

        engine_suffix = f"-bs{max_batch_size}-{'fp16' if fp16_mode else 'fp32'}"
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


        # --- Крок 4: Логіка блокування та конвертації (без змін) ---
        if os.path.exists(trt_engine_path):
             # ... (код перевірки існування та очищення) ...
             print(f"TensorRT двигун вже існує: {trt_engine_path}")
             if os.path.exists(onnx_file_path):
                 try: os.remove(onnx_file_path)
                 except OSError as e: warnings.warn(f"Не вдалося видалити залишки ONNX '{onnx_file_path}': {e}")
             external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
             for f in glob.glob(external_data_pattern):
                 try: os.remove(f)
                 except OSError as e: warnings.warn(f"Не вдалося видалити залишки зовнішніх даних ONNX '{f}': {e}")
             return model_info

        print(f"Цільовий TensorRT двигун: {trt_engine_path}")
        print(f"Файл блокування: {lock_file_path}")
        print(f"Потенційний ONNX файл: {onnx_file_path}")

        lock_acquired = False
        conversion_done = False
        try:
            # --- Логіка отримання блокування ---
            # ... (код блокування без змін) ...
            try:
                with open(lock_file_path, 'x') as lock_file:
                    lock_file.write(f"Locked by PID: {os.getpid()} at {time.ctime()}")
                lock_acquired = True
                print("Блокування встановлено.")
            except FileExistsError:
                print(f"Файл блокування '{lock_file_path}' вже існує. Очікування...")
                start_wait_time = time.monotonic()
                while time.monotonic() - start_wait_time < self.trt_build_wait_timeout:
                    if os.path.exists(trt_engine_path):
                        print("TensorRT двигун з'явився під час очікування.")
                        if os.path.exists(onnx_file_path):
                             try: os.remove(onnx_file_path)
                             except OSError as e: warnings.warn(f"Не вдалося видалити залишки ONNX '{onnx_file_path}' після очікування: {e}")
                        external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                        for f in glob.glob(external_data_pattern):
                             try: os.remove(f)
                             except OSError as e: warnings.warn(f"Не вдалося видалити залишки зовнішніх даних ONNX '{f}' після очікування: {e}")
                        return model_info
                    if not os.path.exists(lock_file_path):
                        print("Файл блокування зник. Спроба отримати блокування...")
                        break
                    time.sleep(self.trt_wait_interval)
                else:
                    if not os.path.exists(trt_engine_path):
                        if os.path.exists(lock_file_path):
                            try: os.remove(lock_file_path)
                            except: pass
                        raise TimeoutError(f"Час очікування ({self.trt_build_wait_timeout}s) вичерпано, двигун '{trt_engine_path}' не з'явився.")
                    else:
                         if os.path.exists(onnx_file_path):
                              try: os.remove(onnx_file_path)
                              except OSError as e: warnings.warn(f"Не вдалося видалити залишки ONNX '{onnx_file_path}' після таймауту: {e}")
                         external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                         for f in glob.glob(external_data_pattern):
                            try: os.remove(f)
                            except OSError as e: warnings.warn(f"Не вдалося видалити залишки зовнішніх даних ONNX '{f}' після таймауту: {e}")
                         return model_info

            if not lock_acquired:
                 try:
                    with open(lock_file_path, 'x') as lock_file:
                        lock_file.write(f"Locked by PID: {os.getpid()} at {time.ctime()} (after wait)")
                    lock_acquired = True
                    print("Блокування успішно встановлено після очікування.")
                 except FileExistsError:
                    raise RuntimeError(f"Не вдалося отримати блокування '{lock_file_path}' після очікування (конкурентний доступ).")
                 except Exception as e_lock:
                    raise RuntimeError(f"Не вдалося отримати блокування '{lock_file_path}' після очікування: {e_lock}") from e_lock
            # --- Кінець логіки блокування ---

            # --- Блок конвертації ---
            if lock_acquired:
                if os.path.exists(trt_engine_path):
                     print(f"Двигун '{trt_engine_path}' вже існує після отримання блокування. Пропуск конвертації.")
                     conversion_done = False
                else:
                    print(f"\n--- Початок конвертації TensorRT (тип: {converter_type}) ---")
                    print(f"GPU: {gpu_name_raw}")
                    print(f"Оригінальна модель (шлях/папка): {original_model_path}")
                    print(f"Цільовий двигун: {trt_engine_path}")
                    print(f"Параметри: max_batch_size={max_batch_size}, fp16={fp16_mode}")

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
                        conversion_done = True
                        print(f"--- Конвертація TensorRT (тип: {converter_type}) успішно завершена ---")

                    except (ImportError, FileNotFoundError, ValueError, RuntimeError, TypeError, KeyError) as e:
                        import traceback
                        traceback.print_exc()
                        if lock_acquired and os.path.exists(lock_file_path):
                            try: os.remove(lock_file_path)
                            except: pass
                        if os.path.exists(onnx_file_path):
                             try: os.remove(onnx_file_path)
                             except: pass
                        raise RuntimeError(f"Помилка під час конвертації TRT (тип: {converter_type}): {e}") from e
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        if lock_acquired and os.path.exists(lock_file_path):
                            try: os.remove(lock_file_path)
                            except: pass
                        if os.path.exists(onnx_file_path):
                             try: os.remove(onnx_file_path)
                             except: pass
                        raise RuntimeError(f"Неочікувана помилка під час конвертації TRT (тип: {converter_type}): {e}") from e
            # --- Кінець блоку конвертації ---

        finally:
            # --- Логіка зняття блокування та очистки (ОНОВЛЕНО v2) ---

            engine_exists = os.path.exists(trt_engine_path)

            # Знімаємо блокування, ЯКЩО воно було встановлено ЦИМ процесом
            if lock_acquired and os.path.exists(lock_file_path):
                try:
                    os.remove(lock_file_path)
                    print("Блокування знято.")
                except OSError as e:
                    warnings.warn(f"Не вдалося видалити файл блокування '{lock_file_path}': {e}")

            # Очищення тимчасових файлів (ONNX та зовнішні дані), ЯКЩО двигун існує
            if engine_exists:
                print(f"Очищення тимчасових файлів у директорії: {trt_target_dir}")
                files_in_target_dir = []
                try:
                    files_in_target_dir = os.listdir(trt_target_dir)
                except OSError as e:
                     warnings.warn(f"Не вдалося отримати список файлів у '{trt_target_dir}': {e}")

                for filename in files_in_target_dir:
                    file_path_to_check = os.path.join(trt_target_dir, filename)
                    # --- ЗМІНА ТУТ: Видаляємо все, що НЕ є .engine файлом ---
                    # (і не є файлом блокування, хоча він вже має бути видалений)
                    if not filename.endswith(".engine") and file_path_to_check != lock_file_path:
                    # --- КІНЕЦЬ ЗМІНИ ---
                        try:
                            if os.path.isfile(file_path_to_check):
                                print(f"  - Видалення тимчасового файлу: {filename}")
                                os.remove(file_path_to_check)
                        except OSError as e:
                            warnings.warn(f"Не вдалося видалити тимчасовий файл '{file_path_to_check}': {e}")

            elif conversion_done: # Якщо конвертували, але двигун НЕ з'явився
                 print(f"Спроба очищення файлів у '{trt_target_dir}' після невдалої конвертації...")
                 files_in_target_dir = []
                 try: files_in_target_dir = os.listdir(trt_target_dir)
                 except OSError as e: warnings.warn(f"Не вдалося отримати список файлів у '{trt_target_dir}': {e}")

                 for filename in files_in_target_dir:
                     file_path_to_check = os.path.join(trt_target_dir, filename)
                     # При невдачі видаляємо все (крім lock, якщо він дивом залишився)
                     if file_path_to_check != lock_file_path:
                         try:
                            if os.path.isfile(file_path_to_check):
                                print(f"  - Видалення залишків: {filename}")
                                os.remove(file_path_to_check)
                         except OSError as e:
                             warnings.warn(f"Не вдалося видалити залишки '{file_path_to_check}': {e}")

        # --- Решта методу (перевірка існування та return) без змін ---
        if not os.path.exists(trt_engine_path):
             raise RuntimeError(f"Не вдалося створити або знайти TensorRT двигун '{trt_engine_path}'. Перевірте логи вище.")
        print(f"Успішно отримано TensorRT двигун: {trt_engine_path}")
        return model_info


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
                                       trt_onnx_opset=17,
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