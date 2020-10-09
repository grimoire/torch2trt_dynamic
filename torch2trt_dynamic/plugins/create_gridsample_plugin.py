import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_gridsample_plugin(layer_name,
                            mode,
                            padding_mode,
                            align_corners):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'GridSamplePluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_mode = trt.PluginField("mode", np.array(
        [mode], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_mode)

    pf_padding_mode = trt.PluginField("padding_mode", np.array(
        [padding_mode], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_padding_mode)

    pf_align_corners = trt.PluginField("align_corners", np.array(
        [align_corners], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_align_corners)

    return creator.create_plugin(layer_name, pfc)