import abc
import glob
import os
import warnings
from typing import Dict, Any, Optional
import tensorrt as trt # Потрібно для TRT_LOGGER

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt else None


def get_network_creation_flags() -> int:
    """
    Флаги для `builder.create_network(...)`, сумісні з різними версіями
    Python-біндінгів TensorRT.

    У деяких релізах (напр. tensorrt==11.2.1.2, на відміну від 10.x) член
    енуму `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` відсутній — explicit
    batch став єдиним підтримуваним режимом і сам флаг став зайвим. Якщо
    атрибут є — повертаємо той самий біт, що й раніше (без зміни поведінки
    для вже робочих версій); якщо відсутній — `0`, що на таких версіях
    еквівалентно (explicit batch і так застосовується за замовчуванням).
    """
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    return 0


def platform_supports_fast(builder: "trt.Builder", attr: str) -> bool:
    """
    Обгортка над `Builder.platform_has_fast_fp16`/`platform_has_fast_int8`,
    сумісна з релізами Python-біндінгів TensorRT, де ці властивості
    прибрані (той самий клас несумісностей, що й
    `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` — див.
    `get_network_creation_flags`). Коли атрибута немає — вважаємо
    підтримку наявною: сучасні TensorRT-білдери підтримують ці режими
    повсюдно, а для шарів, що не вміють, TensorRT сам робить fallback.
    """
    if not hasattr(builder, attr):
        return True
    return bool(getattr(builder, attr))


def try_set_precision_flag(config: "trt.IBuilderConfig", flag_name: str) -> bool:
    """
    Безпечно вмикає `config.set_flag(trt.BuilderFlag.<flag_name>)`.

    У деяких релізах Python-біндінгів TensorRT (спостережено разом із
    відсутністю `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` та
    `Builder.platform_has_fast_fp16` — ймовірно збірка з переходом на
    "strongly typed networks", де точність визначається типами тензорів
    у самому графі, а не глобальними прапорцями білдера) член енуму
    `BuilderFlag.FP16`/`INT8` взагалі відсутній. Якщо його немає —
    попереджаємо й повертаємось до FP32 замість падіння: рушій все одно
    збереться, лише без пришвидшення нижчої точності на цій платформі.

    Повертає True, якщо прапорець дійсно було встановлено.
    """
    if not hasattr(trt.BuilderFlag, flag_name):
        warnings.warn(
            f"trt.BuilderFlag.{flag_name} відсутній у цій версії Python-біндінгів TensorRT "
            f"({getattr(trt, '__version__', '?')}) — продовжую без нього (FP32)."
        )
        return False
    config.set_flag(getattr(trt.BuilderFlag, flag_name))
    return True


def build_engine_from_onnx(onnx_path: str, engine_path: str, fp16_mode: bool = True,
                            max_batch_size: int = 1, memory_limit: Optional[int] = None) -> None:
    """
    Будує статичний TensorRT-двигун з готового ONNX-файлу за допомогою
    "сирого" TensorRT Python API (Builder/OnnxParser), без залежності від
    будь-якого фреймворкового `.export(format="engine")` (`ultralytics` тощо)
    — той самий підхід, що вже використовується в `image_classifier.py`, але
    як окрема, повторно використовувана функція, і з вхідною формою, що
    зчитується напряму з розпарсеної мережі (а не хардкодиться як `"images"`).

    Raises RuntimeError/ValueError з докладним описом на будь-якому кроці
    (парсинг ONNX, побудова, збереження).
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(get_network_creation_flags())
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    if memory_limit:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_limit)
    else:
        default_mem_limit_gb = 2
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, default_mem_limit_gb * (1024 ** 3))
        warnings.warn(f"Ліміт пам'яті для TRT builder не вказано, встановлено {default_mem_limit_gb} GB")

    if fp16_mode:
        if platform_supports_fast(builder, "platform_has_fast_fp16"):
            try_set_precision_flag(config, "FP16")
        else:
            warnings.warn("Платформа не має швидкої підтримки FP16.")

    try:
        # parse_from_file() (not parse(bytes)) -- newer torch.onnx.export
        # writes large weights into a sibling "<name>.onnx.data" external-
        # data file instead of inlining them. parser.parse() only sees the
        # raw bytes with no filesystem context, so it can't resolve that
        # external file and fails with "Failed to import initializer" on
        # whichever weight got externalized. parse_from_file() resolves
        # external data relative to the .onnx file's own directory and
        # works identically to parse() when there is no external data.
        success = parser.parse_from_file(onnx_path)
    except Exception as e:
        raise RuntimeError(f"Помилка читання/парсингу ONNX '{onnx_path}': {e}") from e

    if not success:
        error_msgs = "".join(f"{parser.get_error(i)}\n" for i in range(parser.num_errors))
        external_data_pattern = f"{onnx_path}.data"
        external_files = glob.glob(external_data_pattern) + glob.glob(f"{os.path.splitext(onnx_path)[0]}.*.weight")
        if external_files:
            error_msgs += f"\nПОПЕРЕДЖЕННЯ: Знайдено файли зовнішніх даних ONNX: {external_files}."
        raise RuntimeError(f"Не вдалося розпарсити ONNX файл '{onnx_path}'. Помилки:\n{error_msgs}")

    if network.num_inputs == 0 or network.num_outputs == 0:
        raise RuntimeError(
            f"Мережа TensorRT не має вхідних/вихідних вузлів після парсингу ONNX "
            f"(inputs={network.num_inputs}, outputs={network.num_outputs})."
        )

    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)
    if any(d < 0 for d in input_shape):
        raise ValueError(
            f"ONNX-вхід '{input_tensor.name}' має динамічні виміри {input_shape} — "
            f"цей білдер підтримує лише статичні форми (без optimization profile)."
        )
    if input_shape[0] != max_batch_size:
        raise ValueError(
            f"ONNX-вхід '{input_tensor.name}' має batch-вимір {input_shape[0]}, "
            f"що не збігається з очікуваним max_batch_size={max_batch_size} — "
            f"ONNX мав бути експортований з тим самим batch_size."
        )

    try:
        serialized_engine = None
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
        else:  # старіші версії TensorRT
            engine = builder.build_engine(network, config)
            if engine:
                serialized_engine = engine.serialize()
    except Exception as e:
        raise RuntimeError(f"Помилка під час побудови TRT двигуна: {e}") from e

    if serialized_engine is None:
        raise RuntimeError("Не вдалося побудувати TensorRT двигун (serialized_engine is None).")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    temp_engine_path = engine_path + ".tmp"
    try:
        with open(temp_engine_path, "wb") as f:
            f.write(serialized_engine)
        os.rename(temp_engine_path, engine_path)
    except Exception as e:
        if os.path.exists(temp_engine_path):
            try:
                os.remove(temp_engine_path)
            except OSError:
                pass
        raise RuntimeError(f"Не вдалося зберегти TensorRT двигун: {e}") from e
    finally:
        del serialized_engine, config, parser, network, builder


def trt_export_nms_enabled(model_config: Dict[str, Any]) -> bool:
    """`"tensorrt": {"nms": true}` у конфігу моделі: запікати NMS у граф при експорті
    (ultralytics `export(nms=True)`, вихід (N, max_det, 6) замість сирої сітки)."""
    return bool((model_config.get("tensorrt") or {}).get("nms"))


class BaseTrtConverter(abc.ABC):
    """Абстрактний базовий клас для конвертерів TensorRT."""

    @abc.abstractmethod
    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str], # Деякі конвертери можуть не використовувати ONNX явно
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        """
        Виконує конвертацію моделі у формат TensorRT.

        Args:
            original_model_path: Шлях до оригінального файлу моделі (.pt, .onnx тощо).
            engine_path: Цільовий шлях для збереження .engine файлу.
            onnx_path: Шлях для тимчасового ONNX файлу (якщо використовується).
            model_config: Словник з повною конфігурацією моделі (з JSON).
            builder_config: Словник з параметрами для побудови TRT
                              (напр., 'fp16_mode', 'max_batch_size', 'memory_limit', 'opset').
        """
        pass