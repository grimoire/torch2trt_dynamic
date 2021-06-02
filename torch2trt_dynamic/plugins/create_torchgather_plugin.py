import numpy as np
import tensorrt as trt


def create_torchgather_plugin(layer_name, dim):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchGatherPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_dim = trt.PluginField('dim', np.array([dim], dtype=np.int32),
                             trt.PluginFieldType.INT32)
    pfc.append(pf_dim)

    return creator.create_plugin(layer_name, pfc)
