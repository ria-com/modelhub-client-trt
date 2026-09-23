import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import torch
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    warnings.warn("Бібліотека ultralytics не знайдена. Встановіть її: pip install ultralytics")

from .base import BaseTrtConverter, build_engine_from_onnx, trt_export_nms_enabled, trt_dynamic_hw_enabled


def _is_legacy_yolov5_checkpoint_error(e: Exception) -> bool:
    """
    True, якщо `ultralytics.YOLO(path)` відмовився завантажувати чекпоінт,
    бо це "класичний" YOLOv5/torch.hub чекпоінт (запікльований проти
    власного пакету `models` репозиторію ultralytics/yolov5), а не
    нативний ultralytics (YOLOv8+) чекпоінт. Перевіряємо і саме
    виключення, і весь ланцюжок cause/context — ultralytics зазвичай
    ре-рейзить це як TypeError/RuntimeError із власним повідомленням
    поверх оригінального ModuleNotFoundError.
    """
    needles = ("NOT forwards compatible", "No module named 'models'", "No module named \"models\"")

    def _matches(exc: Optional[BaseException]) -> bool:
        if exc is None:
            return False
        return any(n in str(exc) for n in needles)

    cur: Optional[BaseException] = e
    for _ in range(6):
        if _matches(cur):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if cur is None:
            break
    return False


def _is_missing_trt_api_error(e: Exception) -> bool:
    """
    True, якщо помилка — це `AttributeError` про відсутній атрибут у
    `tensorrt`-модулі/об'єктах (напр. `NetworkDefinitionCreationFlag.
    EXPLICIT_BATCH`, `BuilderFlag.FP16`, `Builder.platform_has_fast_fp16`)
    — ознака того, що ultralytics'ний власний `.export(format="engine")`
    написаний проти іншої версії Python-біндінгів TensorRT, ніж
    встановлена тут. Перевіряє весь ланцюжок cause/context, бо ultralytics
    зазвичай обгортає внутрішні винятки у власний `RuntimeError`.
    """
    def _matches(exc: Optional[BaseException]) -> bool:
        return isinstance(exc, AttributeError) and "tensorrt" in str(exc).lower()

    cur: Optional[BaseException] = e
    for _ in range(6):
        if _matches(cur):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if cur is None:
            break
    return False


def _ensure_yolov5_repo_cached(original_model_path: str) -> str:
    """
    Кешує репозиторій ultralytics/yolov5 (torch.hub) і повертає шлях до нього.

    Виконується в ОКРЕМОМУ підпроцесі: `torch.hub.load("ultralytics/yolov5", ...)`
    імпортує пакети репозиторію з генеричними іменами (`models`, `utils`) прямо
    в `sys.modules`. У довгоживучому процесі це ламає все, що імпортується
    після: наступний YOLOv5-чекпоінт `ultralytics.YOLO()` вже "успішно"
    розпікльовує через чужий `models.yolo`, йде нативним шляхом експорту і
    падає з `BaseModel.fuse() got an unexpected keyword argument 'imgsz'`
    (залежить від порядку конвертацій), а в застосунках, що використовують цей
    пакет, можуть підмінитися їхні власні модулі `models`/`utils`.
    """
    repo_dir = os.path.join(torch.hub.get_dir(), "ultralytics_yolov5_master")
    export_script = os.path.join(repo_dir, "export.py")
    if not os.path.exists(export_script):
        code = (
            "import sys, torch; "
            "torch.hub.load('ultralytics/yolov5', 'custom', path=sys.argv[1], autoshape=False, "
            "trust_repo=True, skip_validation=True, verbose=False, device='cpu')"
        )
        result = subprocess.run([sys.executable, "-c", code, original_model_path],
                                capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(
                f"Не вдалося закешувати репозиторій ultralytics/yolov5 через torch.hub "
                f"(код {result.returncode}).\nSTDERR:\n{result.stderr[-4000:]}"
            )
    if not os.path.exists(export_script):
        raise RuntimeError(
            f"Не знайдено export.py репозиторію ultralytics/yolov5 за очікуваним шляхом: {export_script}"
        )
    return repo_dir


class YoloConverter(BaseTrtConverter):
    """Конвертер TensorRT для моделей YOLO.

    Дві родини чекпоінтів, дві різні стратегії конвертації, і одна
    спільна резервна стратегія для обох, коли встановлена версія
    Python-біндінгів TensorRT несумісна з тим, що очікує `ultralytics`:

    - Нативні ultralytics (YOLOv8+) `.pt` — основний шлях:
      `ultralytics.YOLO(...).export(format="engine")`. Якщо це падає
      через відсутній атрибут `tensorrt`-модуля (інша версія біндінгів,
      напр. без `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`) —
      `_convert_via_onnx_fallback`: ONNX-експорт через той самий
      `ultralytics.YOLO(...).export(format="onnx")` (не торкається
      TensorRT API взагалі), потім побудова двигуна через власний
      `build_engine_from_onnx`.
    - Класичні YOLOv5/torch.hub `.pt` (запікльовані проти пакету `models`
      з репозиторію ultralytics/yolov5) — `ultralytics.YOLO()` їх відверто
      відхиляє ("NOT forwards compatible"); для них окремий шлях:
      `_convert_legacy_yolov5` (ONNX-експорт через власний, підтримуваний
      `export.py` репозиторію yolov5, потім та сама `build_engine_from_onnx`).

    В обох резервних випадках ONNX-експорт лишається на плечах
    відповідного, добре підтримуваного фреймворку (ultralytics або
    репозиторію yolov5) — тут не робиться жодної самописної трасування
    графу моделі; переписаний лише останній крок (ONNX → .engine), який
    і є єдиним місцем, залежним від версії TensorRT.
    """

    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str], # Не використовується прямо, але може бути створено ultralytics
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        """
        Конвертує YOLO модель (.pt) в TensorRT двигун.
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            raise RuntimeError("Конвертація YOLO в TensorRT вимагає CUDA GPU.")

        print(f"(YoloConverter) Завантаження YOLO моделі з: {original_model_path}")
        try:
            model = YOLO(original_model_path)
        except Exception as e:
            if _is_legacy_yolov5_checkpoint_error(e):
                print(
                    "(YoloConverter) Чекпоінт розпізнано як класичний YOLOv5/torch.hub "
                    "(ultralytics.YOLO() його не завантажує) — перемикаюсь на окремий шлях "
                    "конвертації через власний export.py репозиторію ultralytics/yolov5."
                )
                self._convert_legacy_yolov5(
                    original_model_path, engine_path, model_config, builder_config,
                    fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                    imgsz_h=imgsz_h, imgsz_w=imgsz_w,
                )
                return
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Помилка під час конвертації YOLO моделі '{original_model_path}' за допомогою ultralytics: {e}") from e

        try:
            print(f"Початок експорту YOLO в TensorRT (fp16={fp16_mode}, batch={max_batch_size}, imgsz={imgsz})...")

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
            # Без nms=True ultralytics експортує сиру сітку (напр. (1, 5, 8400)),
            # з ним — NMS запечений у граф, вихід (1, max_det, 6). Вмикається
            # лише моделям, яким це потрібно: "tensorrt": {"nms": true} у конфігу.
            if trt_export_nms_enabled(model_config):
                export_args["nms"] = True
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
            try:
                exported_file_path = model.export(**export_args)
            except Exception as e:
                if _is_missing_trt_api_error(e):
                    print(
                        "(YoloConverter) ultralytics.export(format='engine') впав через "
                        "несумісність з цією версією Python-біндінгів TensorRT (відсутній "
                        "атрибут, напр. NetworkDefinitionCreationFlag.EXPLICIT_BATCH) — "
                        "перемикаюсь на резервний шлях: ONNX-експорт через ultralytics, "
                        "потім побудова двигуна через власний, сумісніший TensorRT білдер."
                    )
                    self._convert_via_onnx_fallback(
                        model, engine_path, imgsz=imgsz, device=device,
                        fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                        memory_limit=builder_config.get("memory_limit"),
                    )
                    return
                raise

            print(f"Ultralytics експортував модель у: {exported_file_path}")

            # Перевіряємо, чи експортований файл є .engine
            # (новіші версії ultralytics повертають pathlib.Path замість str)
            if not exported_file_path or not str(exported_file_path).endswith(".engine"):
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

    def _convert_via_onnx_fallback(self, model: "YOLO", engine_path: str, *, imgsz, device: str,
                                    fp16_mode: bool, max_batch_size: int, memory_limit) -> None:
        """
        Резервний шлях для нативних ultralytics (YOLOv8+) чекпоінтів, коли
        `model.export(format="engine")` падає через несумісність
        встановленої версії Python-біндінгів TensorRT з тим, що очікує
        ultralytics (див. `_is_missing_trt_api_error`).

        Робить ONNX-експорт тим самим `model.export(...)` (цей шлях не
        торкається TensorRT API взагалі, лише `torch`/`onnx`), а тоді
        будує двигун через `build_engine_from_onnx` — той самий підхід,
        що й `_convert_legacy_yolov5` та вже робочий `image_classifier.py`.
        """
        onnx_export_args = {"format": "onnx", "device": device, "batch": max_batch_size, "dynamic": False, "half": False}
        if imgsz:
            onnx_export_args["imgsz"] = imgsz

        print(f"(YoloConverter/onnx-fallback) Запуск model.export з аргументами: {onnx_export_args}")
        onnx_path = model.export(**onnx_export_args)
        if not onnx_path or not str(onnx_path).endswith(".onnx") or not os.path.exists(onnx_path):
            raise RuntimeError(
                f"ultralytics ONNX-експорт (резервний шлях) не повернув валідний шлях до .onnx: {onnx_path!r}"
            )

        print(f"(YoloConverter/onnx-fallback) ONNX готовий: {onnx_path}. Побудова TensorRT-двигуна...")
        try:
            build_engine_from_onnx(
                onnx_path, engine_path,
                fp16_mode=fp16_mode, max_batch_size=max_batch_size, memory_limit=memory_limit,
            )
        finally:
            try:
                os.remove(onnx_path)
            except OSError:
                pass
        print(f"(YoloConverter/onnx-fallback) TensorRT-двигун збережено: {engine_path}")

    def _convert_legacy_yolov5(self, original_model_path: str, engine_path: str,
                                model_config: Dict[str, Any], builder_config: Dict[str, Any], *,
                                fp16_mode: bool, max_batch_size: int,
                                imgsz_h: Optional[int], imgsz_w: Optional[int]) -> None:
        """
        Конвертує класичний YOLOv5/torch.hub `.pt` чекпоінт у TensorRT:

        1. ONNX-експорт через ОРИГІНАЛЬНИЙ, підтримуваний `export.py` з
           репозиторію `ultralytics/yolov5` (кешується/клонується через
           `torch.hub`), викликаний в ІЗОЛЬОВАНОМУ підпроцесі — свідомо,
           щоб не забруднювати `sys.modules` цього (довгоживучого,
           спільного для багатьох конвертерів) процесу генерично названими
           пакетами репозиторію (`models`, `utils`), які могли б
           зіткнутися з однойменними пакетами інших моделей/конвертерів.
        2. Побудова TensorRT-двигуна з отриманого ONNX через "сирий"
           TensorRT API (`build_engine_from_onnx`) — той самий підхід, що
           вже працює для `image_classifier.py`, і який НЕ залежить від
           `ultralytics`'ного `.export(format="engine")` (де й сидить
           окрема, вже відома несумісність з деякими версіями `tensorrt`).

        Статична форма входу (фіксований `imgsz`, batch=1), як і в
        `build_engine_from_onnx` (без optimization profiles). З
        `"tensorrt": {"dynamic_hw": true}` ONNX експортується з `--dynamic`, а
        двигун отримує profile з висотою/шириною до `imgsz` — тоді на вхід
        можна подавати той самий прямокутний letterbox, що й torch.hub AutoShape.
        """
        h = imgsz_h or 640
        w = imgsz_w or 640
        opset = builder_config.get("opset", 12)
        dynamic_hw = trt_dynamic_hw_enabled(model_config)

        repo_dir = _ensure_yolov5_repo_cached(original_model_path)
        export_script = os.path.join(repo_dir, "export.py")

        with tempfile.TemporaryDirectory(prefix="yolov5_legacy_export_") as tmp_dir:
            tmp_weights = os.path.join(tmp_dir, os.path.basename(original_model_path))
            shutil.copyfile(original_model_path, tmp_weights)

            cmd = [
                sys.executable, export_script,
                "--weights", tmp_weights,
                "--include", "onnx",
                "--imgsz", str(h), str(w),
                "--batch-size", str(max_batch_size),
                "--opset", str(opset),
                "--device", "0",
            ]
            if dynamic_hw:
                cmd.append("--dynamic")
            print(f"(YoloConverter/legacy-yolov5) Запуск: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                raise RuntimeError(
                    f"export.py репозиторію yolov5 завершився з кодом {result.returncode} "
                    f"для '{original_model_path}'.\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
                )
            print(result.stdout[-2000:])

            onnx_path = os.path.splitext(tmp_weights)[0] + ".onnx"
            if not os.path.exists(onnx_path):
                raise RuntimeError(
                    f"export.py повідомив про успіх, але очікуваний ONNX-файл не знайдено: {onnx_path}\n"
                    f"STDOUT (кінець):\n{result.stdout[-2000:]}"
                )

            print(f"(YoloConverter/legacy-yolov5) ONNX готовий: {onnx_path}. Побудова TensorRT-двигуна...")
            build_engine_from_onnx(
                onnx_path, engine_path,
                fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                memory_limit=builder_config.get("memory_limit"),
                max_hw=(h, w) if dynamic_hw else None,
            )
        print(f"(YoloConverter/legacy-yolov5) TensorRT-двигун збережено: {engine_path}")
