# modelhub_client_trt/trt_converters/__init__.py
from .base import BaseTrtConverter
from .image_classifier import ImageClassifierConverter
from .yolo import YoloConverter
from .text_classifier import TextClassifierConverter # <--- ДОДАНО ІМПОРТ

# Реєстр типів конвертерів
TRT_CONVERTERS = {
    "image_classifier": ImageClassifierConverter,
    "yolo": YoloConverter,
    "text_classifier": TextClassifierConverter, # <--- ДОДАНО ЗАПИС
    # Додайте сюди інші типи конвертерів за потреби
}

# Функція get_converter залишається без змін
def get_converter(converter_type: str) -> BaseTrtConverter:
    """
    Фабрична функція для отримання екземпляра конвертера за типом.
    ... (решта коду без змін) ...
    """
    converter_class = TRT_CONVERTERS.get(converter_type)
    if converter_class:
        return converter_class()
    else:
        raise ValueError(f"Невідомий тип конвертера TensorRT: '{converter_type}'. Доступні: {list(TRT_CONVERTERS.keys())}")

__all__ = ['BaseTrtConverter', 'ImageClassifierConverter', 'YoloConverter', 'TextClassifierConverter', 'get_converter', 'TRT_CONVERTERS'] # <--- ОНОВЛЕНО __all__