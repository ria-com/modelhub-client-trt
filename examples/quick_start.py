import os
import sys
import torch

# add package root
sys.path.append(os.path.join(os.getcwd(), '..'))
from modelhub_client_trt import ModelHubTrt, _TRT_VERSION_STR, get_device_name

# Перевірка доступності CUDA перед запуском прикладу
if not torch.cuda.is_available():
    print("Пропуск прикладу: CUDA недоступний.")
else:
    # Приклад 1: Модель Image Classifier (стандартна логіка)
    model_config_url_classifier = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
    model_name_classifier = "vehicle_registration_certificate_orientation_0_90_detector" # Ця модель має "tensorrt": {"type": "image_classifier"} або відсутня секція

    # Приклад 2: Модель YOLO
    model_config_url_yolo = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/yolo_vrc/model-1.json"
    model_name_yolo = "yolo_vrc_detector" # Ця модель має "tensorrt": {"type": "yolo"}

    local_storage_path = os.environ.get('LOCAL_STORAGE', "../data")
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

    except ImportError as e: print(f"\nПОМИЛКА: Помилка імпорту: {e}.")
    except FileNotFoundError as e: print(f"\nПОМИЛКА: Файл не знайдено - {e}")
    except ValueError as e: print(f"\nПОМИЛКА: Невірні дані - {e}")
    except TimeoutError as e: print(f"\nПОМИЛКА: Таймаут - {e}")
    except RuntimeError as e: print(f"\nПОМИЛКА: Помилка виконання - {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nПОМИЛКА: Неочікувана помилка - {e}"); import traceback; traceback.print_exc()
