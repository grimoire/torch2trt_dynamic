import tensorrt as trt
import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))



def create_meshgrid_plugin(layer_name,
                                num_inputs,
                                slice_dims = [2, 3],
                                starts = [0., 0.],
                                strides = [1., 1.]):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'MeshGridPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_num_inputs = trt.PluginField("num_inputs", np.array(
        [int(num_inputs)], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_num_inputs)

    pf_slice_dims = trt.PluginField("slice_dims", np.array(
        slice_dims, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_slice_dims)

    pf_starts = trt.PluginField("starts", np.array(
        starts, dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_starts)

    pf_strides = trt.PluginField("strides", np.array(
        strides, dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_strides)


    return creator.create_plugin(layer_name, pfc)