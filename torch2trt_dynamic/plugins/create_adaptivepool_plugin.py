import tensorrt as trt
import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))



def create_adaptivepool_plugin(layer_name,
                                output_size,
                                pooling_type):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'AdaptivePoolPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_output_size = trt.PluginField("output_size", np.array(
        output_size, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_output_size)

    pf_pooling_type = trt.PluginField("pooling_type", np.array(
        [int(pooling_type)], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_pooling_type)

    return creator.create_plugin(layer_name, pfc)