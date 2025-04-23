# examples/quick_start.py
import os
import sys
import torch
import time
import tensorrt as trt # <--- Імпорт TRT тут
import warnings # <--- ДОДАНО ІМПОРТ

# add package root
sys.path.append(os.path.join(os.getcwd(), '..'))
from modelhub_client_trt import ModelHubTrt, _TRT_VERSION_STR, get_device_name, _TENSORRT_AVAILABLE

# --- Додаткові імпорти ТІЛЬКИ для перевірки доступності ---
_TEXT_CLASSIFIER_EXAMPLE_AVAILABLE = False # За замовчуванням False
if _TENSORRT_AVAILABLE: # Перевіряємо TRT тут
    try:
        from transformers import AutoTokenizer
        _TEXT_CLASSIFIER_EXAMPLE_AVAILABLE = True # Встановлюємо True, якщо імпорт вдався
        TRT_LOGGER_EXAMPLE = trt.Logger(trt.Logger.WARNING) # Визначаємо логер
    except ImportError:
        print("ПОПЕРЕДЖЕННЯ: Бібліотека 'transformers' недоступна. Приклад TextClassifier буде пропущено.")
    except Exception as e:
        print(f"ПОПЕРЕДЖЕННЯ: Помилка ініціалізації для прикладу TextClassifier: {e}")
else:
    print("ПОПЕРЕДЖЕННЯ: Бібліотека 'tensorrt' недоступна. Приклад TextClassifier буде пропущено.")


# --- Визначаємо функції для прикладу інференсу тут (в глобальній області) ---
def load_trt_engine_example(runtime, engine_path):
    """Завантажує TRT двигун для прикладу."""
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # Можна додати логіку профілів, якщо потрібно, але для простоти поки так
    # profile_index = 0
    # context.set_optimization_profile_async(profile_index=profile_index, stream_handle=torch.cuda.current_stream().cuda_stream)
    return engine, context

def infer_trt_example(context, engine, inputs_dict_torch):
    """Виконує інференс TRT для прикладу."""
    bindings = []
    output_tensors = {}
    stream = torch.cuda.current_stream().cuda_stream

    input_tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    for name in input_tensor_names:
        if name not in inputs_dict_torch:
             raise ValueError(f"Вхідний тензор TRT '{name}' відсутній у наданих даних")
        tensor = inputs_dict_torch[name].contiguous()
        # --- ЗМІНА ТУТ: Використання імпортованого 'warnings' ---
        if tensor.dtype != torch.int32:
             # Тепер warnings.warn спрацює
             warnings.warn(f"Тензор '{name}' має тип {tensor.dtype}, очікується torch.int32. Спроба конвертації.")
             tensor = tensor.to(torch.int32)
        # --- КІНЕЦЬ ЗМІНИ ---
        context.set_input_shape(name, tensor.shape)
        context.set_tensor_address(name, tensor.data_ptr())

    for name in output_tensor_names:
        shape = context.get_tensor_shape(name)
        dtype = torch.float32
        try: dtype = torch.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        except: pass
        output_tensor = torch.empty(tuple(shape), dtype=dtype, device='cuda').contiguous()
        output_tensors[name] = output_tensor
        context.set_tensor_address(name, output_tensor.data_ptr())

    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    return output_tensors
# --- Кінець визначення функцій ---


# --- Основна частина скрипта ---
if not torch.cuda.is_available():
    print("Пропуск прикладу: CUDA недоступний.")
elif not _TENSORRT_AVAILABLE:
     print("Пропуск прикладу: TensorRT недоступний.")
else:
    # ... (визначення URL та імен моделей без змін) ...
    model_config_url_classifier = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
    model_name_classifier = "vehicle_registration_certificate_orientation_0_90_detector"
    model_config_url_yolo = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/yolo_vrc/model-1.json"
    model_name_yolo = "yolo_vrc_detector"
    model_config_url_text_classifier = "https://models.vsp.net.ua/config_model/speech-recognition:support-retrain/xlm-roberta-large-2024-03-24.json"
    model_name_text_classifier = "speech_support"

    local_storage_path = os.environ.get('LOCAL_STORAGE', "../data")
    os.makedirs(local_storage_path, exist_ok=True)
    print(f"Локальне сховище: {local_storage_path}")
    print(f"Версія TensorRT: {_TRT_VERSION_STR}")
    print(f"GPU: {get_device_name()}")

    try:
        modelhub_trt = ModelHubTrt(
            model_config_urls=[
                model_config_url_classifier,
                model_config_url_yolo,
                model_config_url_text_classifier
            ],
            local_storage=local_storage_path,
            trt_build_wait_timeout=600,
            trt_onnx_opset=19,
            trt_builder_memory_limit_gb=8,
            trt_ultralytics_workspace_gb=4
        )

        # --- Тест 1: Image Classifier ---
        print(f"\n=== ТЕСТ 1: Запуск для Image Classifier ({model_name_classifier}) ===")
        # ... (код тесту 1 без змін) ...
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
        from modelhub_client_trt.trt_converters.yolo import _ULTRALYTICS_AVAILABLE
        if _ULTRALYTICS_AVAILABLE:
             print(f"\n=== ТЕСТ 2: Запуск для YOLO ({model_name_yolo}) ===")
             # ... (код тесту 2 без змін) ...
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
        # Перевіряємо прапорець, встановлений на початку
        if _TEXT_CLASSIFIER_EXAMPLE_AVAILABLE:
            print(f"\n=== ТЕСТ 3: Запуск для Text Classifier ({model_name_text_classifier}) ===")
            try:
                optimized_model_info_text = modelhub_trt.download_model_by_name_trt(
                    model_name_text_classifier,
                    max_batch_size=2,
                    fp16_mode=True
                )
                print("\n--- Результат для Text Classifier ---")
                print(f"Оригінальний шлях: {optimized_model_info_text.get('path')}")
                print(f"GPU для оптимізації: {optimized_model_info_text.get('gpu_name')}")
                print(f"Шлях до TRT двигуна: {optimized_model_info_text.get('trt_engine_path')}")
                print(f"Тип конвертера: {optimized_model_info_text.get('tensorrt_converter_type')}")
                print(f"Max Batch Size: {optimized_model_info_text.get('max_batch_size')}")
                print(f"FP16 Mode: {optimized_model_info_text.get('fp16_mode')}")
                print(f"Max Seq Length (з конфігу): {optimized_model_info_text.get('max_seq_length')}")

                # --- Приклад інференсу ---
                trt_engine_path_text = optimized_model_info_text.get('trt_engine_path')
                original_path_text = optimized_model_info_text.get('path')
                max_seq_len_text = optimized_model_info_text.get('max_seq_length')

                if trt_engine_path_text and os.path.exists(trt_engine_path_text) and original_path_text and max_seq_len_text:
                    print("\n--- Приклад інференсу з TRT двигуном Text Classifier ---")
                     # Імпорт AutoTokenizer тут, всередині умови, де він точно доступний
                    from transformers import AutoTokenizer

                    runtime = trt.Runtime(TRT_LOGGER_EXAMPLE) # Використовуємо логер, визначений раніше
                    print(f"Завантаження токенізатора з: {original_path_text}")
                    tokenizer = AutoTokenizer.from_pretrained(original_path_text, use_fast=True)
                    print(f"Завантаження TRT двигуна: {trt_engine_path_text}")
                    # Тепер ці функції визначені глобально
                    engine, context = load_trt_engine_example(runtime, trt_engine_path_text)

                    #test_texts = ["Це приклад нормального тексту."]
                    test_texts = ["Це приклад нормального тексту.", "А тут якась проблема!"]
                    print(f"Токенізація тексту (max_len={max_seq_len_text}): {test_texts}")
                    inputs = tokenizer(
                        test_texts,
                        return_tensors="pt",
                        max_length=max_seq_len_text,
                        padding="max_length",
                        truncation=True
                    )

                    inputs_cuda = {}
                    input_tensor_names_trt = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
                    # print(f"Вхідні тензори, очікувані TRT: {input_tensor_names_trt}") # Дебаг
                    for name, tensor in inputs.items():
                        if name in input_tensor_names_trt:
                             # Переміщуємо на CUDA перед викликом infer_trt_example
                            inputs_cuda[name] = tensor.to(torch.int32).cuda()
                        # else:
                        #    print(f"Попередження: Вхід '{name}' з токенізатора не використовується TRT.")

                    print("Запуск інференсу TRT...")
                    start_time = time.monotonic()
                    # Тепер ці функції визначені глобально
                    trt_outputs = infer_trt_example(context, engine, inputs_cuda)
                    end_time = time.monotonic()
                    print(f"Інференс TRT зайняв: {end_time - start_time:.4f} сек")

                    # Обробка результатів (без змін)
                    output_tensor_names_trt = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
                    # print(f"Вихідні тензори, повернуті TRT: {list(trt_outputs.keys())}")
                    # print(f"Очікувані вихідні тензори TRT: {output_tensor_names_trt}")
                    output_key = None
                    if 'output' in trt_outputs: output_key = 'output'
                    elif output_tensor_names_trt: output_key = output_tensor_names_trt[0]
                    if output_key and output_key in trt_outputs:
                        logits = trt_outputs[output_key]
                        probabilities = torch.softmax(logits, dim=1).cpu().tolist()
                        print("Результати (ймовірності):")
                        for i, text in enumerate(test_texts): print(f"  '{text}': {probabilities[i]}")
                    else: print(f"Не вдалося знайти відповідний вихідний тензор для обробки.")

                    del engine, context, runtime, tokenizer, inputs, inputs_cuda, trt_outputs
                    torch.cuda.empty_cache()

            except Exception as e_text:
                print(f"\nПОМИЛКА під час тестування Text Classifier: {e_text}")
                import traceback
                traceback.print_exc()
        else:
             print("\n=== ТЕСТ 3: Пропущено (Text Classifier) - необхідні бібліотеки недоступні ===")

    except ImportError as e: print(f"\nПОМИЛКА: Помилка імпорту: {e}.")
    except FileNotFoundError as e: print(f"\nПОМИЛКА: Файл не знайдено - {e}")
    except ValueError as e: print(f"\nПОМИЛКА: Невірні дані - {e}")
    except TimeoutError as e: print(f"\nПОМИЛКА: Таймаут - {e}")
    except RuntimeError as e: print(f"\nПОМИЛКА: Помилка виконання - {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nПОМИЛКА: Неочікувана помилка - {e}"); import traceback; traceback.print_exc()