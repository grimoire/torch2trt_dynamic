import tensorrt as trt


def create_torchbmm_plugin(layer_name):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchBmmPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    return creator.create_plugin(layer_name, pfc)
