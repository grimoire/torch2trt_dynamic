import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_torchbmm_plugin(layer_name):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchBmmPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    return creator.create_plugin(layer_name, pfc)