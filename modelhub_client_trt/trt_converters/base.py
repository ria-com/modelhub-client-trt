import abc
from typing import Dict, Any, Optional
import tensorrt as trt # Потрібно для TRT_LOGGER

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt else None

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