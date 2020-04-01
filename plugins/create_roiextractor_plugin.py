import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_roiextractor_plugin(layer_name,
                            out_size,
                            sample_num,
                            featmap_strides,
                            finest_scale):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'RoiExtractorPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_out_size = trt.PluginField("out_size", np.array(
        [out_size], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_out_size)

    pf_sample_num = trt.PluginField("sample_num", np.array(
        [sample_num], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_sample_num)

    pf_featmap_strides = trt.PluginField("featmap_strides", np.array(
        featmap_strides, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_featmap_strides)

    pf_finest_scale = trt.PluginField("finest_scale", np.array(
        [finest_scale], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_finest_scale)

    return creator.create_plugin(layer_name, pfc)