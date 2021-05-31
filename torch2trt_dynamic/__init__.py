import tensorrt as trt

from .converters import *  # noqa: F401,F403
from .torch2trt_dynamic import *  # noqa: F401,F403


def load_plugins():
    import ctypes
    import os
    ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), 'libtorch2trt_dynamic.so'))

    registry = trt.get_plugin_registry()
    torch2trt_creators = [
        c for c in registry.plugin_creator_list
        if c.plugin_namespace == 'torch2trt_dynamic'
    ]
    for c in torch2trt_creators:
        registry.register_creator(c, 'torch2trt_dynamic')


try:
    load_plugins()
    PLUGINS_LOADED = True
except OSError:
    PLUGINS_LOADED = False
