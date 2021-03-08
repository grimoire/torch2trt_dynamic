import numpy as np

import os
import os.path as osp
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_torchembedding_plugin(layer_name, weight):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchEmbeddingPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    num_embeddings = weight.shape[0]
    embedding_dim = weight.shape[1]

    pf_num_embeddings = trt.PluginField(
        "num_embeddings", np.array([num_embeddings], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_num_embeddings)

    pf_embedding_dim = trt.PluginField(
        "embedding_dim", np.array([embedding_dim], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_embedding_dim)

    return creator.create_plugin(layer_name, pfc)