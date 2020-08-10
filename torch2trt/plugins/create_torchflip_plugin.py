import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_torchflip_plugin(layer_name,
                            dims):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchFlipPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_dims = trt.PluginField("dims", np.array(
        dims, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_dims)

    return creator.create_plugin(layer_name, pfc)