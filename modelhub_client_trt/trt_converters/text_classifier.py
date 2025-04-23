# modelhub_client_trt/trt_converters/text_classifier.py
import os
import torch
import onnx
import tensorrt as trt
import warnings
import gc
import logging
from typing import Dict, Any, Optional, List, Tuple

try:
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedTokenizer
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    warnings.warn("Бібліотека 'transformers' не знайдена. TextClassifierConverter не працюватиме.")

from .base import BaseTrtConverter, TRT_LOGGER

# --- Додаємо адаптовану функцію fix_fp16_network ---
def fix_fp16_network(network_definition: trt.INetworkDefinition) -> trt.INetworkDefinition:
    """
    Встановлює FP32 для потенційно проблемних вузлів у FP16 мережі.
    Адаптовано з transformer_deploy.backends.trt_utils.fix_fp16_network
    """
    print("Застосування виправлень для FP16 мережі...")
    count = 0
    # Пошук патернів, які можуть переповнюватися в FP16
    for layer_index in range(network_definition.num_layers - 1):
        layer: trt.ILayer = network_definition.get_layer(layer_index)
        next_layer: trt.ILayer = network_definition.get_layer(layer_index + 1)

        # Знайдено розповсюджений патерн: Pow + ReduceMean (часто після обчислення різниці для MSELoss або подібного)
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # Приведення типів для доступу до атрибутів операцій
            layer.__class__ = trt.IElementWiseLayer
            next_layer.__class__ = trt.IReduceLayer
            # Якщо це операція POW
            if layer.op == trt.ElementWiseOperation.POW:
                # Встановлюємо вищу точність для цих двох шарів
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
                # І для їх виходів
                layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
                next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
                count += 2
                # print(f"  - Встановлено FP32 для POW ({layer.name}) та Reduce ({next_layer.name})")


        # Можна додати інші патерни тут, якщо потрібно, наприклад:
        # if layer.type == trt.LayerType.SOFTMAX:
        #     layer.precision = trt.DataType.FLOAT
        #     layer.set_output_type(0, trt.DataType.FLOAT)
        #     count += 1
        #     print(f"  - Встановлено FP32 для Softmax ({layer.name})")

        # if layer.type == trt.LayerType.LAYER_NORMALIZATION:
        #     layer.precision = trt.DataType.FLOAT
        #     layer.set_output_type(0, trt.DataType.FLOAT)
        #     count += 1
        #     print(f"  - Встановлено FP32 для LayerNorm ({layer.name})")


    if count > 0:
        print(f"Застосовано {count} виправлень точності для FP16.")
    else:
        print("Не знайдено стандартних патернів для виправлення FP16.")
    return network_definition
# --- Кінець функції fix_fp16_network ---


# --- Функції generate_dummy_inputs, convert_hf_to_onnx (без змін відносно попереднього кроку) ---
def generate_dummy_inputs(
    batch_size: int,
    seq_len: int,
    input_names: List[str],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Генерує фіктивні вхідні дані для трасування моделі."""
    shape = (batch_size, seq_len)
    inputs_pytorch: Dict[str, torch.Tensor] = {
        name: torch.ones(size=shape, dtype=torch.int32, device=device) for name in input_names
    }
    return inputs_pytorch

# --- Функція convert_hf_to_onnx (ОНОВЛЕНО) ---
def convert_hf_to_onnx(
    model_pytorch: torch.nn.Module,
    output_path: str,
    inputs_pytorch: Dict[str, torch.Tensor],
    opset_version: int,
    input_names: List[str],
    output_names: List[str] = ["output"],
    var_output_seq: bool = False
):
    """Конвертує модель Hugging Face Pytorch в ONNX."""
    # ... (перевірки та налаштування dynamic_axis без змін) ...
    if not _TRANSFORMERS_AVAILABLE: raise ImportError("Бібліотека 'transformers' недоступна.")
    for k, v in inputs_pytorch.items():
        if not isinstance(v, torch.Tensor): continue
        if v.dtype in [torch.long, torch.int64]: inputs_pytorch[k] = v.type(torch.int32)
        elif v.dtype != torch.int32: warnings.warn(f"Вхід '{k}' тип {v.dtype}.")
    dynamic_axis = dict()
    for k in input_names:
        if k not in inputs_pytorch: continue
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    for output_name in output_names: dynamic_axis[output_name] = {0: "batch_size"}
    use_cache_original = getattr(model_pytorch.config, "use_cache", None)
    if use_cache_original is not None: setattr(model_pytorch.config, "use_cache", False)

    print(f"Експорт в ONNX з параметрами:")
    print(f"  output_path: {output_path}")
    print(f"  opset_version: {opset_version}")
    print(f"  input_names: {input_names}")
    # ... (решта виводу параметрів) ...
    print(f"  Model dtype: {next(model_pytorch.parameters()).dtype}")
    print(f"  Inputs shapes: { {k: v.shape for k, v in inputs_pytorch.items()} }")

    try:
        # --- ЗМІНИ ТУТ: Експорт та перевірка за шляхом ---
        with torch.no_grad():
            torch.onnx.export(
                model_pytorch,
                args=tuple(inputs_pytorch[name] for name in input_names if name in inputs_pytorch),
                f=output_path,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axis,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )
        print(f"ONNX модель успішно збережено: {output_path}")

        # Перевіряємо модель, передаючи шлях до файлу
        print("Перевірка ONNX моделі за шляхом (для великих моделей)...")
        onnx.checker.check_model(output_path)
        print("ONNX модель пройшла перевірку.")

        # Прибираємо спробу завантаження та перезбереження для великих моделей
        # onnx_model = onnx.load(output_path) # <--- ВИДАЛЕНО
        # try: # <--- ВИДАЛЕНО БЛОК
        #     onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        #     print("ONNX модель перезбережено з вбудованими вагами.")
        # except ValueError:
        #     warnings.warn("Не вдалося вбудувати ваги в ONNX (можливо, модель >2GB).")
        # del onnx_model # <--- ВИДАЛЕНО
        # --- КІНЕЦЬ ЗМІН ---

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Помилка під час експорту або перевірки ONNX: {e}") from e
    finally:
        if use_cache_original is not None:
            setattr(model_pytorch.config, "use_cache", use_cache_original)


# --- Функція build_trt_engine (ОНОВЛЕНО) ---
def build_trt_engine(
    onnx_file_path: str,
    engine_file_path: str,
    fp16: bool,
    input_shapes: Dict[str, Dict[str, Tuple[int, int]]],
    workspace_size: int,
    logger: trt.Logger,
):
    """Будує TensorRT двигун з ONNX файлу."""
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # --- ЗМІНИ ТУТ ---
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # Додаємо прапор для кращої відповідності точності FP32
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print("Увімкнено FP16 для TensorRT з OBEY_PRECISION_CONSTRAINTS.")
        else:
            warnings.warn("Платформа не підтримує швидкий FP16.")
    # --- КІНЕЦЬ ЗМІН ---

    print(f"Парсинг ONNX файлу: {onnx_file_path}")
    success = False
    try:
        with open(onnx_file_path, "rb") as model_file:
            success = parser.parse(model_file.read(), path=onnx_file_path)
    except Exception as e:
         raise RuntimeError(f"Помилка читання/парсингу ONNX файлу '{onnx_file_path}': {e}") from e
    if not success:
        error_msgs = ""
        for error in range(parser.num_errors): error_msgs += f"{parser.get_error(error)}\n"
        raise RuntimeError(f"Не вдалося розпарсити ONNX файл '{onnx_file_path}'. Помилки:\n{error_msgs}")
    print("ONNX файл успішно розпарсено.")

    profile = builder.create_optimization_profile()
    num_inputs = network.num_inputs
    if num_inputs == 0: raise RuntimeError("Мережа TensorRT не має вхідних вузлів.")
    network_input_names = [network.get_input(i).name for i in range(num_inputs)]
    print(f"Налаштування профілю оптимізації для входів: {network_input_names}")
    for input_name in network_input_names:
         if input_name not in input_shapes:
              if len(input_shapes) == 1:
                    profile_key = list(input_shapes.keys())[0]
                    warnings.warn(f"Ім'я '{input_name}' не знайдено у формах {list(input_shapes.keys())}. Використовується профіль '{profile_key}'.")
                    shapes = input_shapes[profile_key]
              else:
                  raise ValueError(f"Не знайдено форму для входу '{input_name}'. Надані: {list(input_shapes.keys())}. Входи мережі: {network_input_names}")
         else: shapes = input_shapes[input_name]
         min_s, opt_s, max_s = shapes['min'], shapes['opt'], shapes['max']
         print(f"  - {input_name}: min={min_s}, opt={opt_s}, max={max_s}")
         try: profile.set_shape(input=input_name, min=min_s, opt=opt_s, max=max_s)
         except Exception as e: raise RuntimeError(f"Помилка форми для '{input_name}': {e}") from e
    config.add_optimization_profile(profile)

    # --- ЗМІНИ ТУТ: Застосовуємо фікс для FP16 ---
    if fp16:
        network = fix_fp16_network(network)
    # --- КІНЕЦЬ ЗМІН ---

    print("Побудова TensorRT двигуна (це може зайняти деякий час)...")
    serialized_engine = None
    try:
         if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
         else:
            engine = builder.build_engine(network, config)
            if engine: serialized_engine = engine.serialize()
            else: raise RuntimeError("build_engine повернув None")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Помилка під час побудови TensorRT двигуна: {e}") from e
    if serialized_engine is None: raise RuntimeError("Не вдалося побудувати TensorRT двигун.")

    print(f"Збереження TensorRT двигуна: {engine_file_path}")
    try:
        os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
        with open(engine_file_path, "wb") as f: f.write(serialized_engine)
        print("TensorRT двигун успішно збережено.")
    except Exception as e: raise RuntimeError(f"Не вдалося зберегти TRT двигун '{engine_file_path}': {e}") from e
    finally: del serialized_engine, config, parser, network, builder; gc.collect()


# --- Клас конвертера TextClassifierConverter (ОНОВЛЕНО) ---
class TextClassifierConverter(BaseTrtConverter):
    """
    Конвертер TensorRT для моделей класифікації тексту (Hugging Face Transformers).
    """
    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str],
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        """
        Конвертує HuggingFace модель класифікатора в TensorRT через ONNX.
        """
        if not _TRANSFORMERS_AVAILABLE: raise ImportError("Бібліотека 'transformers' не встановлена.")
        if not onnx_path: raise ValueError("Для TextClassifierConverter потрібен шлях до ONNX файлу.")

        fp16_mode = builder_config.get('fp16_mode', True)
        max_batch_size = builder_config.get('max_batch_size', 1)
        memory_limit = builder_config.get('memory_limit')
        # --- ЗМІНА ТУТ: Використовуємо opset 19 ---
        opset = builder_config.get('opset', 19) # Змінено на 19
        # --- КІНЕЦЬ ЗМІНИ ---

        max_seq_length = model_config.get("max_seq_length")
        if not max_seq_length: raise ValueError("Параметр 'max_seq_length' відсутній.")
        use_fast_tokenizer = model_config.get("use_fast_tokenizer", True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu": raise RuntimeError("Конвертація в TensorRT потребує CUDA GPU.")

        print(f"(TextClassifier) Конвертація: {original_model_path}")
        print(f"Параметри TRT: fp16={fp16_mode}, max_batch={max_batch_size}, max_seq={max_seq_length}, opset={opset}")

        print("Завантаження моделі та токенізатора Hugging Face...")
        try:
            config = AutoConfig.from_pretrained(original_model_path)
            tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(original_model_path, use_fast=use_fast_tokenizer)
            # Завантажуємо модель у FP32
            model = AutoModelForSequenceClassification.from_pretrained(original_model_path, config=config)
            model.eval()
            model.to(device)
            # --- ЗМІНА ТУТ: НЕ переводимо модель в .half() ---
            # if fp16_mode:
            #      model.half() # Закоментовано
            #      print("Модель переведено в режим FP16.") # Закоментовано
            # --- КІНЕЦЬ ЗМІНИ ---
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження з '{original_model_path}': {e}") from e

        input_names = tokenizer.model_input_names
        print(f"Імена входів моделі: {input_names}")

        trace_batch_size = max_batch_size
        trace_seq_len = max_seq_length
        print(f"Генерація фіктивних вхідних даних (batch={trace_batch_size}, seq_len={trace_seq_len})...")
        dummy_input_data = generate_dummy_inputs(
            batch_size=trace_batch_size,
            seq_len=trace_seq_len,
            input_names=input_names,
            device=device
        )

        print(f"Конвертація моделі в ONNX (FP32): {onnx_path}")
        try:
            convert_hf_to_onnx(
                model_pytorch=model, # Передаємо модель у FP32
                output_path=onnx_path,
                inputs_pytorch=dummy_input_data,
                opset_version=opset, # Передаємо opset 19
                input_names=input_names
            )
        except Exception as e:
            raise RuntimeError(f"Помилка під час конвертації в ONNX: {e}") from e
        finally:
             del model, config, tokenizer, dummy_input_data
             gc.collect()
             torch.cuda.empty_cache()
             print("Модель PyTorch та вхідні дані звільнено з пам'яті.")

        print(f"Побудова TensorRT двигуна: {engine_path}")
        min_batch, opt_batch, max_batch = 1, max(1, max_batch_size // 2), max_batch_size
        min_seq, opt_seq, max_seq = 16, max_seq_length, max_seq_length
        trt_input_shapes = {}
        for name in input_names:
             trt_input_shapes[name] = {'min': (min_batch, min_seq), 'opt': (opt_batch, opt_seq), 'max': (max_batch, max_seq)}
        print(f"Форми для профілю TensorRT: {trt_input_shapes}")

        try:
            build_trt_engine(
                onnx_file_path=onnx_path,
                engine_file_path=engine_path,
                fp16=fp16_mode, # TRT Builder сам переведе у FP16, якщо True
                input_shapes=trt_input_shapes,
                workspace_size=memory_limit,
                logger=TRT_LOGGER
            )
        except Exception as e:
            raise RuntimeError(f"Помилка під час побудови TensorRT двигуна: {e}") from e
        finally:
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()