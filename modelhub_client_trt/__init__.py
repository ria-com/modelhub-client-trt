# modelhub_client_trt/__init__.py
"""
The modelhub_client_trt module.
"""
# Додаємо імпорт _TENSORRT_AVAILABLE з основного файлу
from .modelhub_client_trt import ModelHubTrt, _TRT_VERSION_STR, get_device_name, _TENSORRT_AVAILABLE

__version__ = '1.1.1' # Або ваша поточна версія

# Додаємо _TENSORRT_AVAILABLE до __all__
__all__ = (
    '__version__',
    'ModelHubTrt',
    '_TRT_VERSION_STR',
    'get_device_name',
    '_TENSORRT_AVAILABLE' # <--- ДОДАНО
)