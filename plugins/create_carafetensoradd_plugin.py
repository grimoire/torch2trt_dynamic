import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_carafetensoradd_plugin(layer_name):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'CarafeTensorAddPluginDynamic', '1', '')
    
    pfc = trt.PluginFieldCollection()
    
    return creator.create_plugin(layer_name, pfc)
