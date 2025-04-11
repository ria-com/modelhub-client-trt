"""
The modelhub_client_trt module.
"""
from .modelhub_client_trt import ModelHubTrt, _TRT_VERSION_STR, get_device_name

__version__ = '1.0.0'

__all__ = (
    '__version__',
    'ModelHubTrt',
    '_TRT_VERSION_STR',
    'get_device_name'
)
