import numpy as np

# import pyamirstan_plugin as pyamir

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_layernorm_plugin(layer_name,
                            normalized_shape,
                            W,
                            B,
                            eps=1e-5,
                            type_id=trt.DataType.FLOAT):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'LayerNormPluginDynamic', '1', '')
    
    pfc = trt.PluginFieldCollection()

    pf_normalized_shape = trt.PluginField("normalized_shape", np.array(
        normalized_shape, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_normalized_shape)

    pf_eps = trt.PluginField("eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_eps)

    pf_W = trt.PluginField("W", W, trt.PluginFieldType.FLOAT32)
    pfc.append(pf_W)

    pf_B = trt.PluginField("B", B, trt.PluginFieldType.FLOAT32)
    pfc.append(pf_B)
    
    pf_type_id = trt.PluginField("type_id", np.array(
        [type_id], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_type_id)
    
    return creator.create_plugin(layer_name, pfc)
