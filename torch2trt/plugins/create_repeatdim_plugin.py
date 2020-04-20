import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_repeat_plugin(layer_name,
                            repeat_shape,
                            type_id=trt.DataType.FLOAT):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'RepeatDimsPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_repeat_shape = trt.PluginField("repeat_dims", np.array(
        repeat_shape, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_repeat_shape)

    pf_type_id = trt.PluginField("type_id", np.array(
        [type_id], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_type_id)

    return creator.create_plugin(layer_name, pfc)