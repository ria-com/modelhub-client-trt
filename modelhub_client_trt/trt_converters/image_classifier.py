import os
import torch
import onnx
import tensorrt as trt
import warnings
import glob
from typing import Dict, Any, Optional

from .base import BaseTrtConverter, TRT_LOGGER

class ImageClassifierConverter(BaseTrtConverter):
    """
    Конвертер TensorRT для моделей класифікації зображень через ONNX.
    Це реалізація за замовчуванням.
    """
    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str],
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        """
        Конвертує TorchScript модель класифікатора в TensorRT через ONNX.
        """
        if not onnx_path:
            raise ValueError("Для ImageClassifierConverter потрібен шлях до ONNX файлу.")

        fp16_mode = builder_config.get('fp16_mode', True)
        max_batch_size = builder_config.get('max_batch_size', 1)
        memory_limit = builder_config.get('memory_limit')
        opset = builder_config.get('opset', 13)

        # --- 5. Завантажити модель TorchScript ---
        print(f"(ImageClassifier) Завантаження TorchScript моделі з: {original_model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.jit.load(original_model_path, map_location=device).eval()
        except Exception as jit_err:
            warnings.warn(f"Не вдалося завантажити TorchScript модель '{original_model_path}': {e}")
            try:
                obj = torch.load(original_model_path, map_location=device, weights_only=False)    
                if isinstance(obj, torch.nn.Module):
                    model = obj.to(device)
                else:
                    raise RuntimeError(f"Непідтримуваний тип об’єкта з torch.load: {type(obj)}")
                model.eval()
            except Exception as load_err:
                raise RuntimeError(
                    f"Не вдалося завантажити модель '{original_model_path}' ані як TorchScript, "
                    f"ані через torch.load: {load_err}"
                ) from load_err
            
        if fp16_mode:
            if device.type == 'cuda':
                print("Конвертація завантаженої моделі в FP16...")
                model = model.half()
            else:
                warnings.warn("FP16 mode requested but running on CPU.")

        # --- 6. Підготувати вхідні дані ---
        input_size = model_config.get("input_size")
        image_h = model_config.get("image_size_h")
        image_w = model_config.get("image_size_w")
        if not input_size and (not image_h or not image_w):
            raise ValueError(f"Відсутні 'image_size_h'/'image_size_w' або input_size у конфігурації для моделі")
        if not input_size:
            input_size = (3, image_h, image_w)
        try:
            dtype = torch.half if fp16_mode and device.type == 'cuda' else torch.float
            example_input = torch.randn(max_batch_size, *input_size, device=device, dtype=dtype)
            print(f"Використовується приклад вхідних даних для ONNX ({example_input.dtype}) з формою: {example_input.shape} на {device}")
        except Exception as e:
            raise RuntimeError(f"Помилка створення прикладу вхідних даних: {e}") from e

        # --- 7. Експорт моделі в ONNX ---
        print(f"Експорт в ONNX (opset {opset}) у файл: {onnx_path}...")
        input_names = ['images']
        output_names = ['output']
        onnx_initial_export_succeeded = False
        onnx_model = None
        try:
            # Переконуємось, що директорія для ONNX існує
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            torch.onnx.export(model, example_input, onnx_path, export_params=True, opset_version=opset,
                              do_constant_folding=True, input_names=input_names, output_names=output_names,
                              dynamic_axes=None, verbose=False)
            print("Початковий експорт в ONNX успішний.")
            onnx_initial_export_succeeded = True

            print(f"Перевірка створеної ONNX моделі: {onnx_path}")
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX модель пройшла перевірку.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            if not onnx_initial_export_succeeded:
                raise RuntimeError(f"Помилка під час початкового експорту в ONNX: {e}") from e
            else:
                warnings.warn(f"Помилка під час перевірки ONNX моделі: {e}. Спроба продовжити...")
                if onnx_model is None and os.path.exists(onnx_path):
                    try:
                       onnx_model = onnx.load(onnx_path)
                    except Exception as load_e:
                       warnings.warn(f"Не вдалося завантажити ONNX модель після помилки перевірки: {load_e}")
                       # Якщо не можемо навіть завантажити, то експорт вважається неуспішним
                       onnx_initial_export_succeeded = False

        if onnx_initial_export_succeeded and onnx_model:
            try:
                print(f"Спроба перезбереження ONNX для вбудовування ваг: {onnx_path}")
                onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
                print("ONNX модель перезбережено.")
                if os.path.exists(onnx_path):
                    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                    print(f"Розмір ONNX файлу після перезбереження: {onnx_size_mb:.2f} MB")
                external_data_pattern = f"{os.path.splitext(onnx_path)[0]}.*.weight"
                external_files = glob.glob(external_data_pattern)
                if external_files:
                    warnings.warn(f"Зовнішні дані ONNX ({external_files}) все ще існують!")
            except ValueError as e:
                warnings.warn(f"Не вдалося перезберегти ONNX з внутрішніми вагами (модель >2GB?): {e}. Продовження...")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Помилка під час перезбереження ONNX моделі: {e}") from e
        elif onnx_initial_export_succeeded:
            warnings.warn("Перезбереження ONNX пропущено через помилку перевірки.")

        if not os.path.exists(onnx_path):
             raise RuntimeError(f"Файл ONNX '{onnx_path}' не було створено або його видалено передчасно.")


        # --- 8. Побудова TRT двигуна ---
        print(f"Побудова TensorRT двигуна (fp16={fp16_mode}, max_batch={max_batch_size}) з ONNX...")
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config = builder.create_builder_config()
        if memory_limit:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_limit)
        else:
             # Встановлюємо ліміт за замовчуванням, якщо не передано
             default_mem_limit_gb = 2
             config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, default_mem_limit_gb * (1024**3))
             warnings.warn(f"Ліміт пам'яті для TRT builder не вказано, встановлено {default_mem_limit_gb} GB")


        if fp16_mode:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Увімкнено режим FP16 для білдера TensorRT.")
            else:
                warnings.warn("Платформа не має швидкої підтримки FP16.")

        success = False
        try:
            with open(onnx_path, 'rb') as onnx_model_file:
                print(f"Парсинг ONNX файлу: {onnx_path}")
                success = parser.parse(onnx_model_file.read())
        except Exception as e:
            raise RuntimeError(f"Помилка читання/парсингу ONNX '{onnx_path}': {e}") from e

        if not success:
            error_msgs = ""
            for error in range(parser.num_errors):
                error_msgs += f"{parser.get_error(error)}\n"
            external_data_pattern = f"{os.path.splitext(onnx_path)[0]}.*.weight"
            external_files = glob.glob(external_data_pattern)
            if external_files:
                error_msgs += f"\nПОПЕРЕДЖЕННЯ: Знайдено файли зовнішніх даних ONNX: {external_files}."
            raise RuntimeError(f"Не вдалося розпарсити ONNX файл '{onnx_path}'. Помилки:\n{error_msgs}")
        print("ONNX файл успішно розпарсено.")

        num_inputs = network.num_inputs
        num_outputs = network.num_outputs
        print(f"DEBUG: TensorRT network. Inputs: {num_inputs}, Outputs: {num_outputs}")
        for i in range(num_inputs): inp = network.get_input(i); print(f"DEBUG: TRT Input {i}: Name='{inp.name}', Shape={inp.shape}, DType={inp.dtype}")
        for i in range(num_outputs): out = network.get_output(i); print(f"DEBUG: TRT Output {i}: Name='{out.name}', Shape={out.shape}, DType={out.dtype}")

        if network.num_outputs == 0:
             raise RuntimeError("Мережа TensorRT не має вихідних вузлів після парсингу ONNX.")

        print("Побудова серіалізованого TensorRT двигуна...")
        serialized_engine = None
        try:
            # Встановлення профілю оптимізації для фіксованого батчу
            profile = builder.create_optimization_profile()
            # Припускаємо, що вхід називається 'images', як визначено вище
            input_name = 'images'
            min_shape = (max_batch_size, 3, image_h, image_w) # Використовуємо max_batch_size для всіх
            opt_shape = (max_batch_size, 3, image_h, image_w)
            max_shape = (max_batch_size, 3, image_h, image_w)
            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

            if hasattr(builder, "build_serialized_network"):
                serialized_engine = builder.build_serialized_network(network, config)
            else: # Старіші версії TRT
                 print("Використання build_engine...")
                 engine = builder.build_engine(network, config)
                 if engine:
                      serialized_engine = engine.serialize()
                 else:
                      raise RuntimeError("build_engine повернув None")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Помилка під час побудови TRT двигуна: {e}") from e

        if serialized_engine is None:
            raise RuntimeError("Не вдалося побудувати TensorRT двигун (serialized_engine is None).")
        print("TensorRT двигун успішно побудовано.")

        # --- 9. Збереження двигуна ---
        print(f"Збереження TensorRT двигуна в: {engine_path}")
        try:
            # Переконуємось, що директорія існує
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            temp_engine_path = engine_path + ".tmp"
            with open(temp_engine_path, "wb") as f:
                f.write(serialized_engine)
            os.rename(temp_engine_path, engine_path)
            print("TensorRT двигун успішно збережено.")
        except Exception as e:
            if 'temp_engine_path' in locals() and os.path.exists(temp_engine_path):
                try: os.remove(temp_engine_path)
                except OSError: pass
            raise RuntimeError(f"Не вдалося зберегти TensorRT двигун: {e}") from e
        finally:
            # Явне видалення об'єктів TRT для звільнення пам'яті
            del serialized_engine, config, parser, network, builder
            if 'model' in locals(): del model # Звільнення моделі з пам'яті GPU/CPU
            if 'example_input' in locals(): del example_input
            if torch.cuda.is_available(): torch.cuda.empty_cache()
