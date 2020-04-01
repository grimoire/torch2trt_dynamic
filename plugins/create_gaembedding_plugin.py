import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_gaembedding_plugin(layer_name,
                            position_magnitude,
                            q_stride,
                            kv_stride,
                            feat_dim,
                            wave_length=1000,
                            type_id=trt.DataType.FLOAT):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'GeneralizedAttentionEmbeddingPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_position_magnitude = trt.PluginField("position_magnitude", np.array(
        [position_magnitude], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_position_magnitude)

    pf_q_stride = trt.PluginField("q_stride", np.array(
        [q_stride], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_q_stride)

    pf_kv_stride = trt.PluginField("kv_stride", np.array(
        [kv_stride], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_kv_stride)

    pf_feat_dim = trt.PluginField("feat_dim", np.array(
        [feat_dim], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_feat_dim)

    pf_wave_length = trt.PluginField("wave_length", np.array(
        [wave_length], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_wave_length)

    pf_type_id = trt.PluginField("type_id", np.array(
        [type_id], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_type_id)

    return creator.create_plugin(layer_name, pfc)