from .base import BaseTrtConverter
from .image_classifier import ImageClassifierConverter
from .yolo import YoloConverter

# Реєстр типів конвертерів
TRT_CONVERTERS = {
    "image_classifier": ImageClassifierConverter,
    "yolo": YoloConverter,
    # Додайте сюди інші типи конвертерів за потреби
}

def get_converter(converter_type: str) -> BaseTrtConverter:
    """
    Фабрична функція для отримання екземпляра конвертера за типом.

    Args:
        converter_type: Рядок, що ідентифікує тип конвертера (напр., "image_classifier", "yolo").

    Returns:
        Екземпляр відповідного класу конвертера.

    Raises:
        ValueError: Якщо вказаний тип конвертера не знайдено.
    """
    converter_class = TRT_CONVERTERS.get(converter_type)
    if converter_class:
        return converter_class()
    else:
        raise ValueError(f"Невідомий тип конвертера TensorRT: '{converter_type}'. Доступні: {list(TRT_CONVERTERS.keys())}")

__all__ = ['BaseTrtConverter', 'ImageClassifierConverter', 'YoloConverter', 'get_converter', 'TRT_CONVERTERS']