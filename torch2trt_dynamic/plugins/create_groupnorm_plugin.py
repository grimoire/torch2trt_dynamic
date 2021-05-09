import numpy as np
import tensorrt as trt


def create_groupnorm_plugin(layer_name, num_groups, eps=1e-5):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'GroupNormPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_num_groups = trt.PluginField("num_groups",
                                    np.array([num_groups], dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    pfc.append(pf_num_groups)

    pf_eps = trt.PluginField("eps", np.array([eps], dtype=np.float32),
                             trt.PluginFieldType.FLOAT32)
    pfc.append(pf_eps)

    return creator.create_plugin(layer_name, pfc)
