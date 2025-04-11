import os
import shutil
import warnings
import torch
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    warnings.warn("Бібліотека ultralytics не знайдена. Встановіть її: pip install ultralytics")

from .base import BaseTrtConverter

class YoloConverter(BaseTrtConverter):
    """Конвертер TensorRT для моделей YOLO з використанням ultralytics."""

    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str], # Не використовується прямо, але може бути створено ultralytics
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        """
        Конвертує YOLO модель (.pt) в TensorRT двигун за допомогою ultralytics.
        """
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError("Бібліотека ultralytics не встановлена, неможливо конвертувати YOLO модель.")

        if not original_model_path.endswith(".pt"):
            warnings.warn(f"Очікувався файл моделі .pt для YOLO, отримано: {original_model_path}")

        fp16_mode = builder_config.get('fp16_mode', True)
        max_batch_size = builder_config.get('max_batch_size', 1)
        # Ultralytics може використовувати інші параметри для розміру зображення,
        # але ми можемо їх передати, якщо API export це підтримує
        imgsz_h = model_config.get("image_size_h")
        imgsz_w = model_config.get("image_size_w")
        imgsz = max(imgsz_h, imgsz_w) if imgsz_h and imgsz_w else None # YOLO часто використовує один розмір

        if imgsz is None:
            warnings.warn("Розміри зображення (image_size_h/w) не вказані в конфігу, ultralytics може використати розмір за замовчуванням.")


        print(f"(YoloConverter) Завантаження YOLO моделі з: {original_model_path}")
        try:
            model = YOLO(original_model_path)
            print(f"Початок експорту YOLO в TensorRT (fp16={fp16_mode}, batch={max_batch_size}, imgsz={imgsz})...")

            # Визначаємо пристрій
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cpu':
                 raise RuntimeError("Конвертація YOLO в TensorRT вимагає CUDA GPU.")


            # Викликаємо експорт з ultralytics
            # workspace=4 (GB) - рекомендовано для TRT експорту в ultralytics
            # half=fp16_mode
            # batch=max_batch_size (динамічний батч за замовчуванням, це може бути max batch)
            # device=device
            # imgsz=imgsz (якщо визначено)
            export_args = {
                 "format": "engine",
                 "half": fp16_mode,
                 "batch": max_batch_size, # Передаємо max_batch_size сюди
                 "device": device,
                 "workspace": builder_config.get('ultralytics_workspace_gb', 4),
            }
            if imgsz:
                export_args["imgsz"] = imgsz

            # Шлях, куди ultralytics збереже двигун за замовчуванням
            # Зазвичай це <original_name>.engine в тій самій директорії
            default_engine_name = f"{os.path.splitext(os.path.basename(original_model_path))[0]}.engine"
            # Визначаємо тимчасову директорію, де будемо запускати export,
            # щоб контролювати, де створюється файл .engine
            temp_export_dir = os.path.dirname(engine_path) # Використовуємо цільову директорію
            # Важливо: ultralytics може створити файл .engine в поточній робочій директорії
            # або в директорії моделі. Краще копіювати оригінальну модель у цільову
            # директорію і запускати експорт звідти, або вказати 'project' і 'name'
            # Або просто перемістити результат

            # Переконаємось, що цільова директорія існує
            os.makedirs(temp_export_dir, exist_ok=True)

            print(f"Запуск model.export з аргументами: {export_args}")
            # model.export повертає шлях до експортованого файлу
            exported_file_path = model.export(**export_args)

            print(f"Ultralytics експортував модель у: {exported_file_path}")

            # Перевіряємо, чи експортований файл є .engine
            if not exported_file_path or not exported_file_path.endswith(".engine"):
                 # Можливо, export повернув шлях до .onnx, якщо були проблеми з TRT
                 # Або щось інше пішло не так
                 # Спробуємо знайти .engine файл поруч з оригінальним файлом
                 expected_default_path = os.path.join(os.path.dirname(original_model_path), default_engine_name)
                 if os.path.exists(expected_default_path):
                      print(f"Знайдено .engine файл за замовчуванням: {expected_default_path}")
                      exported_file_path = expected_default_path
                 else:
                      # Спробуємо знайти в цільовій директорії
                       expected_target_path = os.path.join(temp_export_dir, default_engine_name)
                       if os.path.exists(expected_target_path):
                            print(f"Знайдено .engine файл у цільовій директорії: {expected_target_path}")
                            exported_file_path = expected_target_path
                       else:
                            # Спробуємо знайти .onnx, якщо TRT не вдався
                             default_onnx_name = f"{os.path.splitext(os.path.basename(original_model_path))[0]}.onnx"
                             expected_onnx_path = os.path.join(temp_export_dir, default_onnx_name)
                             if os.path.exists(expected_onnx_path):
                                  warnings.warn(f"Експорт у .engine, схоже, не вдався. Знайдено .onnx файл: {expected_onnx_path}. Подальша конвертація не виконується цим конвертером.")
                                  # Ми не можемо продовжити, бо очікували .engine
                                  raise RuntimeError(f"Ultralytics export не повернув шлях до .engine файлу і він не був знайдений. Знайдено можливий ONNX: {expected_onnx_path}")
                             else:
                                 raise RuntimeError(f"Ultralytics export не повернув шлях до .engine файлу, і він не був знайдений за стандартними шляхами ({expected_default_path}, {expected_target_path}).")


            # Переміщення/перейменування файлу, якщо він не там, де треба
            if os.path.abspath(exported_file_path) != os.path.abspath(engine_path):
                print(f"Переміщення/перейменування з '{exported_file_path}' у '{engine_path}'")
                # Переконуємось, що цільовий файл не існує (на випадок повторного запуску)
                if os.path.exists(engine_path):
                    os.remove(engine_path)
                shutil.move(exported_file_path, engine_path)
                print(f"Файл успішно переміщено/перейменовано в: {engine_path}")
            else:
                print(f"TensorRT двигун вже знаходиться за цільовим шляхом: {engine_path}")

            # Очистка можливих проміжних файлів (напр., onnx), які міг створити ultralytics
            # у директорії ВИХІДНОЇ моделі.
            print("(YoloConverter) Перевірка наявності проміжних файлів у вихідній директорії...")
            source_model_dir = os.path.dirname(original_model_path)
            base_name_no_ext = os.path.splitext(os.path.basename(original_model_path))[0]

            # Формуємо шлях до можливого .onnx файлу у вихідній директорії
            possible_onnx_in_source = os.path.join(source_model_dir, f"{base_name_no_ext}.onnx")

            if os.path.exists(possible_onnx_in_source):
                # Перевіряємо, чи onnx_path (якщо передано з основного класу) вказує на цей файл.
                # Це малоймовірно для YoloConverter, але для безпеки перевіримо.
                if onnx_path and os.path.abspath(possible_onnx_in_source) == os.path.abspath(onnx_path):
                     print(f"(YoloConverter) Проміжний ONNX файл '{possible_onnx_in_source}' керується зовнішньою логікою.")
                else:
                     try:
                          print(f"(YoloConverter) Видалення проміжного файлу: {possible_onnx_in_source}")
                          os.remove(possible_onnx_in_source)
                     except OSError as e:
                          warnings.warn(f"(YoloConverter) Не вдалося видалити проміжний файл '{possible_onnx_in_source}': {e}")
            else:
                 print(f"(YoloConverter) Проміжний ONNX файл '{possible_onnx_in_source}' не знайдено.")

            # Додатково можна шукати інші файли, наприклад .json
            possible_json_in_source = os.path.join(source_model_dir, f"{base_name_no_ext}.json")
            if os.path.exists(possible_json_in_source):
                try:
                    print(f"(YoloConverter) Видалення проміжного файлу: {possible_json_in_source}")
                    os.remove(possible_json_in_source)
                except OSError as e:
                    warnings.warn(f"(YoloConverter) Не вдалося видалити проміжний файл '{possible_json_in_source}': {e}")

        except ImportError as e:
             raise e # Перекидаємо помилку імпорту
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Помилка під час конвертації YOLO моделі '{original_model_path}' за допомогою ultralytics: {e}") from e
        finally:
            if 'model' in locals(): del model # Звільняємо модель
            if torch.cuda.is_available(): torch.cuda.empty_cache()