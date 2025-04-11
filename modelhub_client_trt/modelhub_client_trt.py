# --- Додати імпорти на початку файлу ---
import os
import torch
import warnings
import time
from typing import Dict, List, Optional, Any # Додано Any
import glob
import shutil # Додано для можливого використання в очистці

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
    _TRT_VERSION_STR = f"{trt.__version__.split('.')[0]}.{trt.__version__.split('.')[1]}"
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

# --- Імпорт фабрики конвертерів ---
from .trt_converters import get_converter, BaseTrtConverter # Додано BaseTrtConverter

from .trt_converters.yolo import _ULTRALYTICS_AVAILABLE # Імпортуємо змінну

# --- Функції get_device_name, sanitize_for_filename (без змін) ---
def get_device_name() -> str:
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception as e: warnings.warn(f"Не вдалося отримати назву GPU: {e}"); return "unknown_gpu"
    return "cpu"

def sanitize_for_filename(name: str) -> str:
    name = name.replace(" ", "_")
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars: name = name.replace(char, '')
    name = name.replace("(", "").replace(")", "")
    return name
# --- Кінець незмінної частини ---

class ModelHubTrt(ModelHub):
    """
    Клас для управління моделями з підтримкою TensorRT оптимізації
    з використанням різних стратегій конвертації.
    """
    def __init__(self,
                 models: Optional[Dict[str, Dict[str, str]]] = None,
                 model_config_urls: Optional[List[str]] = None,
                 local_storage: Optional[str] = None,
                 remote_storage: Optional[str] = None,
                 postfix: str = "./modelhub",
                 trt_build_wait_timeout: int = 300, # <--- ПОВЕРНУЛИ ПАРАМЕТР
                 trt_wait_interval: int = 5,       # <--- ПОВЕРНУЛИ ПОВ'ЯЗАНИЙ ПАРАМЕТР
                 trt_onnx_opset: int = 13,
                 trt_builder_memory_limit_gb: int = 2,
                 trt_ultralytics_workspace_gb: int = 4
                 ) -> None:
        super().__init__(models=models, model_config_urls=model_config_urls, local_storage=local_storage,
                         remote_storage=remote_storage, postfix=postfix)
        self.trt_build_wait_timeout = trt_build_wait_timeout # <--- ПОВЕРНУЛИ ІНІЦІАЛІЗАЦІЮ
        self.trt_wait_interval = trt_wait_interval       # <--- ПОВЕРНУЛИ ІНІЦІАЛІЗАЦІЮ
        self.trt_onnx_opset = trt_onnx_opset
        self.trt_builder_memory_limit = trt_builder_memory_limit_gb * (1024 ** 3)
        self.trt_ultralytics_workspace_gb = trt_ultralytics_workspace_gb
        if not _TENSORRT_AVAILABLE:
            warnings.warn("tensorrt не доступний. Функціонал TRT буде обмежено.")
            # ONNX може бути потрібен для деяких конвертерів, але не для всіх
            if not _ONNX_AVAILABLE:
                warnings.warn("onnx не доступний. Деякі типи конвертації TRT можуть не працювати.")

    def download_model_by_name_trt(self,
                                   model_name: str,
                                   path: Optional[str] = None,
                                   max_batch_size: int = 1,
                                   fp16_mode: bool = True) -> Dict[str, str]: # Додав fp16_mode сюди
        """
        Завантажує модель та конвертує її в TensorRT .engine формат,
        використовуючи стратегію, визначену в конфігурації моделі.

        Args:
            model_name: Назва моделі для завантаження та конвертації.
            path: Локальний шлях для збереження (якщо відрізняється від налаштувань за замовчуванням).
            max_batch_size: Максимальний розмір батчу для TensorRT двигуна.
            fp16_mode: Чи використовувати FP16 точність при побудові двигуна.

        Returns:
            Словник з інформацією про модель, включаючи шлях до TRT двигуна.

        Raises:
            ImportError: Якщо необхідні бібліотеки (tensorrt, onnx, ultralytics) відсутні для обраного типу конвертації.
            FileNotFoundError: Якщо оригінальний файл моделі не знайдено.
            ValueError: Якщо конфігурація моделі неповна або некоректна.
            RuntimeError: Якщо виникають помилки під час конвертації.
            TimeoutError: Якщо час очікування блокування вичерпано.
        """
        # Перевірка доступності TensorRT на початку
        if not _TENSORRT_AVAILABLE:
             raise ImportError("Бібліотека tensorrt не встановлена або не ініціалізована.")

        # --- Кроки 1-3: Завантаження, конфіг, шляхи (майже без змін) ---
        print(f"Завантаження оригінальної моделі '{model_name}'...")
        model_info = super().download_model_by_name(model_name, path=path)
        original_model_path = model_info.get("path")
        if not original_model_path or not os.path.exists(original_model_path):
            raise FileNotFoundError(f"Оригінальний файл моделі не знайдено: {original_model_path}")
        print(f"Оригінальна модель завантажена: {original_model_path}")

        model_config = self.models.get(model_name)
        if not model_config:
            raise ValueError(f"Конфігурацію моделі не знайдено для '{model_name}'")

        # --- Визначення типу конвертера ---
        tensorrt_config = model_config.get("tensorrt", {}) # Отримуємо секцію tensorrt
        converter_type = tensorrt_config.get("type", "image_classifier") # Тип конвертера, за замовчуванням 'image_classifier'
        print(f"Визначено тип конвертера TensorRT: '{converter_type}'")

        # --- Перевірка GPU та формування шляхів (залишається) ---
        gpu_name_raw = get_device_name()
        if gpu_name_raw == "cpu":
            raise RuntimeError("Оптимізація TensorRT потребує CUDA-сумісного GPU.")
        sanitized_gpu_name = sanitize_for_filename(gpu_name_raw)
        trt_version_sanitized = sanitize_for_filename(_TRT_VERSION_STR)
        original_basename = os.path.basename(original_model_path)
        original_name_part, _ = os.path.splitext(original_basename)

        # Ім'я файлу двигуна тепер включає max_batch_size та fp16 статус для унікальності
        engine_suffix = f"-bs{max_batch_size}-{'fp16' if fp16_mode else 'fp32'}"
        engine_file_name = (f"{original_name_part}-{sanitized_gpu_name}-trt{trt_version_sanitized}{engine_suffix}.engine")

        original_model_dir = os.path.dirname(original_model_path)
        # Створюємо піддиректорію для TRT файлів на основі GPU
        trt_model_dir = os.path.join(original_model_dir, sanitized_gpu_name)
        os.makedirs(trt_model_dir, exist_ok=True)

        trt_engine_path = os.path.join(trt_model_dir, engine_file_name)
        lock_file_path = trt_engine_path + ".lock"
        # ONNX шлях визначається тут, навіть якщо не всі конвертери його використовують,
        # це потрібно для логіки очищення та потенційного використання image_classifier
        onnx_file_path = os.path.join(trt_model_dir, f"{original_name_part}-temp{engine_suffix}.onnx")

        model_info["trt_engine_path"] = trt_engine_path
        model_info["gpu_name"] = gpu_name_raw
        model_info["tensorrt_converter_type"] = converter_type # Додаємо тип конвертера в інфо
        model_info["max_batch_size"] = max_batch_size
        model_info["fp16_mode"] = fp16_mode
        # --- Кінець Кроків 1-3 ---

        # --- Крок 4: Логіка блокування (без змін, тільки використовує оновлені шляхи) ---
        if os.path.exists(trt_engine_path):
            print(f"TensorRT двигун вже існує: {trt_engine_path}")
            # Очистка тимчасового ONNX, якщо він є
            if os.path.exists(onnx_file_path):
                 try: os.remove(onnx_file_path)
                 except OSError: pass
            # Очистка можливих зовнішніх даних ONNX (рідко, але можливо)
            external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
            for f in glob.glob(external_data_pattern):
                 try: os.remove(f)
                 except OSError: pass
            return model_info

        print(f"Цільовий TensorRT двигун: {trt_engine_path}")
        print(f"Файл блокування: {lock_file_path}")
        print(f"Потенційний ONNX файл: {onnx_file_path}") # Змінено на "потенційний"

        lock_acquired = False
        conversion_done = False # Прапорець, що конвертація була запущена
        try:
            # --- Логіка отримання блокування (без змін) ---
            try:
                # 'x' - ексклюзивне створення, fail якщо існує
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
                        # Очистка ONNX якщо він залишився від попереднього процесу
                        if os.path.exists(onnx_file_path):
                             try: os.remove(onnx_file_path)
                             except OSError: pass
                        external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                        for f in glob.glob(external_data_pattern):
                             try: os.remove(f)
                             except OSError: pass
                        return model_info # Успішно дочекалися
                    if not os.path.exists(lock_file_path):
                        print("Файл блокування зник. Спроба отримати блокування...")
                        break # Виходимо з циклу очікування, щоб спробувати захопити
                    time.sleep(self.trt_wait_interval)
                else: # while завершився за таймаутом
                    if not os.path.exists(trt_engine_path):
                        raise TimeoutError(f"Час очікування ({self.trt_build_wait_timeout}s) вичерпано, двигун '{trt_engine_path}' не з'явився.")
                    else:
                        # Двигун з'явився в останній момент
                         if os.path.exists(onnx_file_path):
                              try: os.remove(onnx_file_path)
                              except OSError: pass
                         external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                         for f in glob.glob(external_data_pattern):
                            try: os.remove(f)
                            except OSError: pass
                         return model_info

            # Якщо вийшли з циклу очікування через зникнення lock-файлу, спроба захоплення
            if not lock_acquired:
                 try:
                    with open(lock_file_path, 'x') as lock_file:
                        lock_file.write(f"Locked by PID: {os.getpid()} at {time.ctime()} (after wait)")
                    lock_acquired = True
                    print("Блокування успішно встановлено після очікування.")
                 except FileExistsError:
                    # Це може статися, якщо інший процес встиг створити lock між перевірками
                    raise RuntimeError(f"Не вдалося отримати блокування '{lock_file_path}' після очікування (конкурентний доступ).")

            # --- Кінець логіки блокування ---


            # --- Блок конвертації з використанням стратегії ---
            if lock_acquired:
                # Перевірка, чи файл не був створений іншим процесом, поки ми отримували лок
                if os.path.exists(trt_engine_path):
                     print(f"Двигун '{trt_engine_path}' вже існує після отримання блокування. Пропуск конвертації.")
                     conversion_done = False # Не ми конвертували
                else:
                    print(f"\n--- Початок конвертації TensorRT (тип: {converter_type}) ---")
                    print(f"GPU: {gpu_name_raw}")
                    print(f"Оригінальна модель: {original_model_path}")
                    print(f"Цільовий двигун: {trt_engine_path}")
                    print(f"Параметри: max_batch_size={max_batch_size}, fp16={fp16_mode}")

                    try:
                        # Отримуємо екземпляр конвертера
                        converter: BaseTrtConverter = get_converter(converter_type)

                        # Готуємо конфігурацію для конвертера
                        builder_config = {
                            'fp16_mode': fp16_mode,
                            'max_batch_size': max_batch_size,
                            'memory_limit': self.trt_builder_memory_limit,
                            'opset': self.trt_onnx_opset, # Потрібно для image_classifier
                            'ultralytics_workspace_gb': self.trt_ultralytics_workspace_gb, # Потрібно для yolo
                        }

                        # Викликаємо метод конвертації
                        converter.convert(
                            original_model_path=original_model_path,
                            engine_path=trt_engine_path,
                            onnx_path=onnx_file_path, # Передаємо шлях, навіть якщо конвертер його не використовує
                            model_config=model_config,
                            builder_config=builder_config
                        )
                        conversion_done = True # Позначаємо, що конвертація була ініційована нами
                        print(f"--- Конвертація TensorRT (тип: {converter_type}) успішно завершена ---")

                    except (ImportError, FileNotFoundError, ValueError, RuntimeError, TypeError) as e:
                        # Ловимо можливі помилки з конвертерів
                        import traceback
                        traceback.print_exc()
                        # Залишаємо lock файл, щоб інші процеси не намагалися повторити невдалу конвертацію?
                        # Або краще видалити, щоб дозволити спробу повторно? Видалимо.
                        raise RuntimeError(f"Помилка під час конвертації TRT (тип: {converter_type}): {e}") from e
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Неочікувана помилка під час конвертації TRT (тип: {converter_type}): {e}") from e

            # --- Кінець блоку конвертації ---

        finally:
            # --- Логіка зняття блокування та очистки ---
            if lock_acquired and os.path.exists(lock_file_path):
                try:
                    os.remove(lock_file_path)
                    print("Блокування знято.")
                except OSError as e:
                    warnings.warn(f"Не вдалося видалити файл блокування '{lock_file_path}': {e}")

            # Очищаємо тимчасовий ONNX файл, якщо він був створений (або мав бути створений)
            # і якщо двигун був успішно створений (або вже існував)
            if os.path.exists(trt_engine_path):
                 if os.path.exists(onnx_file_path):
                    try:
                        os.remove(onnx_file_path)
                        print(f"Тимчасовий ONNX файл '{onnx_file_path}' видалено.")
                    except OSError as e:
                        warnings.warn(f"Не вдалося видалити ONNX файл '{onnx_file_path}': {e}")

                 # Очистка можливих зовнішніх даних ONNX
                 external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                 external_files = glob.glob(external_data_pattern)
                 if external_files:
                      print(f"Видалення зовнішніх даних ONNX: {external_files}")
                      for f in external_files:
                           try:
                                os.remove(f)
                           except OSError as e:
                                warnings.warn(f"Не вдалося видалити зовнішні дані ONNX '{f}': {e}")
            elif conversion_done:
                 # Якщо ми намагались конвертувати, але двигун не з'явився,
                 # то теж варто прибрати можливі залишки ONNX
                 if os.path.exists(onnx_file_path):
                     try:
                         os.remove(onnx_file_path)
                         print(f"Видалення залишків ONNX '{onnx_file_path}' після невдалої конвертації.")
                     except OSError as e:
                         warnings.warn(f"Не вдалося видалити залишки ONNX '{onnx_file_path}': {e}")


        # Фінальна перевірка наявності двигуна
        if not os.path.exists(trt_engine_path):
             # Якщо двигун не існує після всіх маніпуляцій, це помилка
             raise RuntimeError(f"Не вдалося створити або знайти TensorRT двигун '{trt_engine_path}'. Перевірте логи вище.")

        print(f"Успішно отримано TensorRT двигун: {trt_engine_path}")
        return model_info

# --- Приклад використання __main__ ---
if __name__ == '__main__':
    # Перевірка доступності CUDA перед запуском прикладу
    if not torch.cuda.is_available():
        print("Пропуск прикладу: CUDA недоступний.")
    elif not _TENSORRT_AVAILABLE:
         print("Пропуск прикладу: TensorRT недоступний.")
    else:
        # Приклад 1: Модель Image Classifier (стандартна логіка)
        model_config_url_classifier = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
        model_name_classifier = "vehicle_registration_certificate_orientation_0_90_detector" # Ця модель має "tensorrt": {"type": "image_classifier"} або відсутня секція

        # Приклад 2: Модель YOLO
        model_config_url_yolo = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/yolo_vrc/model-1.json"
        model_name_yolo = "yolo_vrc_detector" # Ця модель має "tensorrt": {"type": "yolo"}

        local_storage_path = os.environ.get('LOCAL_STORAGE', "./downloaded_models_trt_refactored")
        os.makedirs(local_storage_path, exist_ok=True)
        print(f"Локальне сховище: {local_storage_path}")
        print(f"Версія TensorRT: {_TRT_VERSION_STR}")
        print(f"GPU: {get_device_name()}")

        try:
            modelhub_trt = ModelHubTrt(model_config_urls=[model_config_url_classifier, model_config_url_yolo],
                                       local_storage=local_storage_path,
                                       trt_build_wait_timeout=600, # Збільшено таймаут
                                       trt_onnx_opset=19,
                                       trt_builder_memory_limit_gb=4,
                                       trt_ultralytics_workspace_gb=4) # Ліміт для YOLO

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
            # Перевірка, чи доступний ultralytics
            if _ULTRALYTICS_AVAILABLE:
                print(f"\n=== ТЕСТ 2: Запуск для YOLO ({model_name_yolo}) ===")
                optimized_model_info_yolo = modelhub_trt.download_model_by_name_trt(
                    model_name_yolo,
                    max_batch_size=1, # Спробуємо з батчем 1
                    fp16_mode=True   # і FP16
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


        except ImportError as e: print(f"\nПОМИЛКА: Помилка імпорту: {e}.")
        except FileNotFoundError as e: print(f"\nПОМИЛКА: Файл не знайдено - {e}")
        except ValueError as e: print(f"\nПОМИЛКА: Невірні дані - {e}")
        except TimeoutError as e: print(f"\nПОМИЛКА: Таймаут - {e}")
        except RuntimeError as e: print(f"\nПОМИЛКА: Помилка виконання - {e}"); import traceback; traceback.print_exc()
        except Exception as e: print(f"\nПОМИЛКА: Неочікувана помилка - {e}"); import traceback; traceback.print_exc()