import os
import torch
import warnings
import time
from typing import Dict, List, Optional
import glob

# --- Імпорти та глобальні змінні (onnx, tensorrt, TRT_LOGGER etc.) ---
# (Залишаються з попередньої версії)
try:
    import onnx
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    warnings.warn("Бібліотека onnx не знайдена. Встановіть її: pip install onnx")

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
    Клас для управління моделями... (Додано ONNX checker та логування виходів TRT)
    """
    # --- __init__ (без змін) ---
    def __init__(self,
                 models: Optional[Dict[str, Dict[str, str]]] = None,
                 model_config_urls: Optional[List[str]] = None,
                 local_storage: Optional[str] = None,
                 remote_storage: Optional[str] = None,
                 postfix: str = "./modelhub",
                 trt_build_wait_timeout: int = 300,
                 trt_wait_interval: int = 5,
                 trt_onnx_opset: int = 13,
                 trt_builder_memory_limit_gb: int = 2
                 ) -> None:
        super().__init__(models=models, model_config_urls=model_config_urls, local_storage=local_storage,
                         remote_storage=remote_storage, postfix=postfix)
        self.trt_build_wait_timeout = trt_build_wait_timeout
        self.trt_wait_interval = trt_wait_interval
        self.trt_onnx_opset = trt_onnx_opset
        self.trt_builder_memory_limit = trt_builder_memory_limit_gb * (1024 ** 3)
        if not _ONNX_AVAILABLE or not _TENSORRT_AVAILABLE:
            warnings.warn("onnx та/або tensorrt не доступні.")
    # --- Кінець __init__ ---

    def download_model_by_name_trt(self,
                                   model_name: str,
                                   path: Optional[str] = None,
                                   max_batch_size: int = 1) -> Dict[str, str]:
        fp16_mode = True
        if not _ONNX_AVAILABLE or not _TENSORRT_AVAILABLE:
            raise ImportError("Бібліотеки onnx та/або tensorrt не встановлені.")

        # --- Кроки 1-3: Завантаження, конфіг, шляхи (без змін) ---
        print(f"Завантаження оригінальної моделі '{model_name}'...")
        model_info = super().download_model_by_name(model_name, path=path)
        original_model_path = model_info.get("path")
        if not original_model_path or not os.path.exists(original_model_path):
            raise FileNotFoundError(f"Оригінальний файл моделі не знайдено: {original_model_path}")
        print(f"Оригінальна модель завантажена: {original_model_path}")
        model_config = self.models.get(model_name)
        if not model_config: raise ValueError(f"Конфігурацію моделі не знайдено для '{model_name}'")
        gpu_name_raw = get_device_name()
        if gpu_name_raw == "cpu": raise RuntimeError("Оптимізація TensorRT потребує CUDA-сумісного GPU.")
        sanitized_gpu_name = sanitize_for_filename(gpu_name_raw)
        trt_version_sanitized = sanitize_for_filename(_TRT_VERSION_STR)
        original_basename = os.path.basename(original_model_path)
        original_name_part, _ = os.path.splitext(original_basename)
        engine_file_name = (f"{original_name_part}-{sanitized_gpu_name}-trt{trt_version_sanitized}.engine")
        original_model_dir = os.path.dirname(original_model_path)
        trt_model_dir = os.path.join(original_model_dir, sanitized_gpu_name)
        os.makedirs(trt_model_dir, exist_ok=True)
        trt_engine_path = os.path.join(trt_model_dir, engine_file_name)
        lock_file_path = trt_engine_path + ".lock"
        onnx_file_path = os.path.join(trt_model_dir, f"{original_name_part}-temp.onnx")
        model_info["trt_engine_path"] = trt_engine_path
        model_info["gpu_name"] = gpu_name_raw
        # --- Кінець Кроків 1-3 ---

        # --- Крок 4: Логіка блокування (без змін, лише додано очистку external data) ---
        if os.path.exists(trt_engine_path):
            print(f"TensorRT двигун вже існує: {trt_engine_path}")
            if os.path.exists(onnx_file_path):
                 try: os.remove(onnx_file_path)
                 except OSError: pass
            external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
            for f in glob.glob(external_data_pattern):
                 try: os.remove(f)
                 except OSError: pass
            return model_info

        print(f"Цільовий TensorRT двигун: {trt_engine_path}")
        print(f"Файл блокування: {lock_file_path}")
        print(f"Тимчасовий ONNX файл: {onnx_file_path}")

        lock_acquired = False
        onnx_export_done = False
        try:
            # --- Отримання блокування (без змін) ---
            try:
                with open(lock_file_path, 'x') as lock_file: lock_file.write(f"Locked by PID: {os.getpid()} at {time.ctime()}")
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
                             except OSError: pass
                        external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"; [os.remove(f) for f in glob.glob(external_data_pattern) if os.path.exists(f)] # One-liner cleanup
                        return model_info
                    if not os.path.exists(lock_file_path):
                        print("Файл блокування зник. Спроба отримати блокування...")
                        break
                    time.sleep(self.trt_wait_interval)
                else:
                    if not os.path.exists(trt_engine_path): raise TimeoutError(f"Час очікування ({self.trt_build_wait_timeout}s) вичерпано, двигун не з'явився.")
                    else:
                        if os.path.exists(onnx_file_path):
                             try: os.remove(onnx_file_path)
                             except OSError: pass
                        external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"; [os.remove(f) for f in glob.glob(external_data_pattern) if os.path.exists(f)]
                        return model_info

            if not lock_acquired:
                 try:
                    with open(lock_file_path, 'x') as lock_file: lock_file.write(f"Locked by PID: {os.getpid()} at {time.ctime()} (after wait)")
                    lock_acquired = True
                    print("Блокування успішно встановлено після очікування.")
                 except FileExistsError: raise RuntimeError(f"Не вдалося отримати блокування '{lock_file_path}' після очікування.")
            # --- Кінець логіки блокування ---


            # --- Блок експорту та побудови ---
            if lock_acquired:
                print(f"Початок експорту в ONNX та побудови TensorRT двигуна для GPU: {gpu_name_raw}")

                # 5. Завантажити модель TorchScript
                print(f"Завантаження TorchScript моделі з: {original_model_path}")
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = torch.jit.load(original_model_path, map_location=device).eval()
                    # Конвертація в FP16 (перевірте параметр fp16_mode!)
                    if fp16_mode:
                        if device.type == 'cuda': print("Конвертація завантаженої моделі в FP16..."); model = model.half()
                        else: warnings.warn("FP16 mode requested but running on CPU.")
                except Exception as e: raise RuntimeError(f"Не вдалося завантажити TorchScript модель '{original_model_path}': {e}") from e

                # 6. Підготувати вхідні дані
                image_h = model_config.get("image_size_h"); image_w = model_config.get("image_size_w")
                if not image_h or not image_w: raise ValueError(f"Відсутні 'image_size_h'/'image_size_w' у конфігурації для '{model_name}'")
                try:
                    dtype = torch.half if fp16_mode and device.type == 'cuda' else torch.float
                    print(f"DEBUG: Перед створенням example_input: fp16_mode={fp16_mode}, device.type={device.type}, selected dtype={dtype}")
                    example_input = torch.randn(max_batch_size, 3, image_h, image_w, device=device, dtype=dtype)
                    print(f"Використовується приклад вхідних даних для ONNX ({example_input.dtype}) з формою: {example_input.shape} на {device}")
                except Exception as e: raise RuntimeError(f"Помилка створення прикладу вхідних даних: {e}") from e

                # 7. Експорт моделі в ONNX
                print(f"Експорт в ONNX (opset {self.trt_onnx_opset}) у файл: {onnx_file_path}...")
                input_names = ['images']; output_names = ['output']
                onnx_initial_export_succeeded = False; onnx_model = None
                try:
                    torch.onnx.export(model, example_input, onnx_file_path, export_params=True, opset_version=self.trt_onnx_opset,
                                      do_constant_folding=True, input_names=input_names, output_names=output_names,
                                      dynamic_axes=None, verbose=False)
                    print("Початковий експорт в ONNX успішний.")
                    onnx_initial_export_succeeded = True

                    # *** НОВЕ: Перевірка моделі ONNX ***
                    print(f"Перевірка створеної ONNX моделі: {onnx_file_path}")
                    onnx_model = onnx.load(onnx_file_path)
                    onnx.checker.check_model(onnx_model)
                    print("ONNX модель пройшла перевірку.")

                except Exception as e:
                     import traceback
                     traceback.print_exc()
                     # Якщо помилка сталася під час перевірки, початковий експорт все ще міг вдатися
                     if not onnx_initial_export_succeeded:
                         raise RuntimeError(f"Помилка під час початкового експорту в ONNX: {e}") from e
                     else:
                         warnings.warn(f"Помилка під час перевірки ONNX моделі: {e}. Спроба продовжити...")
                         # Модель була експортована, але не пройшла перевірку. Продовжуємо на свій ризик.
                         if onnx_model is None: # Завантажуємо, якщо ще не завантажили
                              onnx_model = onnx.load(onnx_file_path)


                # Спроба перезбереження ONNX (залишається з попередньої версії)
                if onnx_initial_export_succeeded and onnx_model: # Перезберігаємо тільки якщо початковий експорт був і модель завантажена
                    try:
                        print(f"Спроба перезбереження ONNX для вбудовування ваг: {onnx_file_path}")
                        onnx.save_model(onnx_model, onnx_file_path, save_as_external_data=False)
                        print("ONNX модель перезбережено.")
                        onnx_export_done = True
                        if os.path.exists(onnx_file_path):
                             onnx_size_mb = os.path.getsize(onnx_file_path) / (1024 * 1024)
                             print(f"Розмір ONNX файлу після перезбереження: {onnx_size_mb:.2f} MB")
                             external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                             external_files = glob.glob(external_data_pattern)
                             if external_files: warnings.warn(f"Зовнішні дані ONNX ({external_files}) все ще існують!")
                    except ValueError as e:
                         warnings.warn(f"Не вдалося перезберегти ONNX з внутрішніми вагами (модель >2GB?): {e}. Продовження...")
                         onnx_export_done = True # Вважаємо експорт завершеним, хоч і з зовнішніми даними
                    except Exception as e:
                         import traceback
                         traceback.print_exc()
                         raise RuntimeError(f"Помилка під час перезбереження ONNX моделі: {e}") from e
                elif onnx_initial_export_succeeded:
                    # Якщо перевірка ONNX не вдалася, але експорт був, ми не перезберігаємо,
                    # але позначимо, що файл існує для подальшої обробки/видалення
                    onnx_export_done = True
                    warnings.warn("Перезбереження ONNX пропущено через помилку перевірки.")


                # 8. Побудова TRT двигуна
                if onnx_export_done:
                    print(f"Побудова TensorRT двигуна (v{_TRT_VERSION_STR}, fp16={fp16_mode}, max_batch={max_batch_size}) з ONNX...")
                    builder = trt.Builder(TRT_LOGGER)
                    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    network = builder.create_network(network_flags)
                    parser = trt.OnnxParser(network, TRT_LOGGER)
                    config = builder.create_builder_config()
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.trt_builder_memory_limit)

                    if fp16_mode:
                        if builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16); print("Увімкнено режим FP16 для білдера TensorRT.")
                        else: warnings.warn("Платформа не має швидкої підтримки FP16.")

                    success = False
                    try:
                        with open(onnx_file_path, 'rb') as onnx_model_file:
                            print(f"Парсинг ONNX файлу: {onnx_file_path}")
                            success = parser.parse(onnx_model_file.read())
                    except Exception as e: raise RuntimeError(f"Помилка читання/парсингу ONNX '{onnx_file_path}': {e}") from e

                    if not success:
                        error_msgs = ""; [error_msgs := error_msgs + f"{parser.get_error(error)}\n" for error in range(parser.num_errors)]
                        external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"; external_files = glob.glob(external_data_pattern)
                        if external_files: error_msgs += f"\nПОПЕРЕДЖЕННЯ: Знайдено файли зовнішніх даних ONNX: {external_files}."
                        raise RuntimeError(f"Не вдалося розпарсити ONNX файл '{onnx_file_path}'. Помилки:\n{error_msgs}")
                    print("ONNX файл успішно розпарсено.")

                    # *** НОВЕ: Логування входів/виходів мережі TRT ***
                    num_inputs = network.num_inputs
                    num_outputs = network.num_outputs
                    print(f"DEBUG: TensorRT network. Inputs: {num_inputs}, Outputs: {num_outputs}")
                    for i in range(num_inputs):
                        inp = network.get_input(i)
                        print(f"DEBUG: TRT Input {i}: Name='{inp.name}', Shape={inp.shape}, DType={inp.dtype}")
                    for i in range(num_outputs):
                        out = network.get_output(i)
                        print(f"DEBUG: TRT Output {i}: Name='{out.name}', Shape={out.shape}, DType={out.dtype}")

                    # Явна перевірка перед побудовою
                    if network.num_outputs == 0:
                        raise RuntimeError("Мережа TensorRT не має вихідних вузлів після парсингу ONNX. Неможливо побудувати двигун.")
                    # --- Кінець логування ---

                    print("Побудова з фіксованим розміром батчу.")
                    print("Побудова серіалізованого TensorRT двигуна...")
                    serialized_engine = None
                    try:
                        if hasattr(builder, "build_serialized_network"): serialized_engine = builder.build_serialized_network(network, config)
                        else: raise RuntimeError("Не знайдено методу build_serialized_network.")
                    except Exception as e: import traceback; traceback.print_exc(); raise RuntimeError(f"Помилка під час побудови TRT двигуна: {e}") from e

                    if serialized_engine is None: raise RuntimeError("Не вдалося побудувати TensorRT двигун (None).")
                    print("TensorRT двигун успішно побудовано.")

                    # 9. Збереження двигуна (без змін)
                    print(f"Збереження TensorRT двигуна в: {trt_engine_path}")
                    try:
                        temp_engine_path = trt_engine_path + ".tmp"
                        with open(temp_engine_path, "wb") as f: f.write(serialized_engine)
                        os.rename(temp_engine_path, trt_engine_path)
                        print("TensorRT двигун успішно збережено.")
                    except Exception as e:
                        if 'temp_engine_path' in locals() and os.path.exists(temp_engine_path):
                             try: os.remove(temp_engine_path)
                             except OSError: pass
                        raise RuntimeError(f"Не вдалося зберегти TensorRT двигун: {e}") from e
                    finally: del serialized_engine, config, parser, network, builder

            # --- Кінець блоку експорту та побудови ---

        finally:
            # --- Логіка зняття блокування та очистки ONNX/зовнішніх даних (без змін) ---
            if lock_acquired and os.path.exists(lock_file_path):
                try: os.remove(lock_file_path); print("Блокування знято.")
                except OSError as e: warnings.warn(f"Не вдалося видалити файл блокування '{lock_file_path}': {e}")
            if onnx_export_done:
                 if os.path.exists(onnx_file_path):
                    try: os.remove(onnx_file_path); print(f"Тимчасовий ONNX файл '{onnx_file_path}' видалено.")
                    except OSError as e: warnings.warn(f"Не вдалося видалити ONNX файл '{onnx_file_path}': {e}")
                 external_data_pattern = f"{os.path.splitext(onnx_file_path)[0]}.*.weight"
                 for f in glob.glob(external_data_pattern):
                      try: print(f"Видалення зовнішніх даних ONNX: {f}"); os.remove(f)
                      except OSError as e: warnings.warn(f"Не вдалося видалити зовнішні дані ONNX '{f}': {e}")

        if not os.path.exists(trt_engine_path):
             raise RuntimeError(f"Не вдалося створити або знайти TensorRT двигун '{trt_engine_path}'.")

        return model_info

# --- Приклад використання __main__ (без змін) ---
if __name__ == '__main__':
    # ... (код прикладу залишається тим самим) ...
    if not _ONNX_AVAILABLE or not _TENSORRT_AVAILABLE or not torch.cuda.is_available():
        print("Пропуск прикладу: onnx, tensorrt не встановлено або CUDA недоступний.")
    else:
        # ... (решта налаштувань прикладу) ...
        model_config_url = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
        model_name_to_optimize = "vehicle_registration_certificate_orientation_0_90_detector"
        local_storage_path = os.environ.get('LOCAL_STORAGE', "./downloaded_models_trt_onnx_check") # Нова папка
        os.makedirs(local_storage_path, exist_ok=True)
        print(f"Локальне сховище: {local_storage_path}")
        print(f"Версія TensorRT: {_TRT_VERSION_STR}")
        print(f"GPU: {get_device_name()}")

        try:
            modelhub_trt = ModelHubTrt(model_config_urls=[model_config_url],
                                       local_storage=local_storage_path,
                                       trt_build_wait_timeout=600,
                                       trt_onnx_opset=13,
                                       trt_builder_memory_limit_gb=4) # Можливо, знадобиться більше пам'яті для великого ONNX

            # *** ПЕРЕВІРТЕ ЦЕЙ ПАРАМЕТР У ВАШОМУ quick_start.py ***
            print(f"\nЗапуск завантаження та TRT оптимізації (через ONNX для моделі: {model_name_to_optimize}")
            optimized_model_info = modelhub_trt.download_model_by_name_trt(
                model_name_to_optimize,
                max_batch_size=1
            )
            # ... (решта коду прикладу) ...
            print("\n--- Результат ---")
            print(f"Оригінальний шлях: {optimized_model_info.get('path')}")
            print(f"GPU для оптимізації: {optimized_model_info.get('gpu_name')}")
            print(f"Шлях до TRT двигуна: {optimized_model_info.get('trt_engine_path')}")

        except ImportError as e: print(f"Помилка імпорту: {e}.")
        except FileNotFoundError as e: print(f"Помилка: Файл не знайдено - {e}")
        except ValueError as e: print(f"Помилка: Невірні дані - {e}")
        except TimeoutError as e: print(f"Помилка: Таймаут - {e}")
        except RuntimeError as e: print(f"Помилка виконання: {e}"); import traceback; traceback.print_exc()
        except Exception as e: print(f"Неочікувана помилка: {e}"); import traceback; traceback.print_exc()