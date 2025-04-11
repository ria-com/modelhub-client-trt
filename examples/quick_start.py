import os
import sys

# add package root
sys.path.append(os.path.join(os.getcwd(), '..'))
from modelhub_client_trt import ModelHubTrt, _TRT_VERSION_STR, get_device_name

# ... (решта налаштувань прикладу) ...
model_config_url = "https://models.vsp.net.ua/config_model/auto-badcontent-detector/vehicle_registration_certificate/ua/orientation/vrc-orientation-0-90-src/model-5.json"
model_name_to_optimize = "vehicle_registration_certificate_orientation_0_90_detector"
local_storage_path = os.environ.get('LOCAL_STORAGE', "../data") # Нова папка
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
    print(f"\nЗапуск завантаження та TRT оптимізації (через ONNX, FP16={run_fp16_mode}) для моделі: {model_name_to_optimize}")
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