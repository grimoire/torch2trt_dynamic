import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_exview_plugin(layer_name,
                            expr_list):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'ExViewPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    expr_str = ';'.join(expr_list)
    pf_dim_expression = trt.PluginField("dim_expression", np.array(
        [ord(i) for i in list(expr_str)], np.uint8), trt.PluginFieldType.CHAR)
    pfc.append(pf_dim_expression)

    return creator.create_plugin(layer_name, pfc)
