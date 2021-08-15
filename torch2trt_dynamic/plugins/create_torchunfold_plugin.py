import numpy as np
import tensorrt as trt


def create_torchunfold_plugin(layer_name, kernel_size, dilation, padding,
                              stride):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchUnfoldPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    pf_kernel_size = trt.PluginField('kernel_size',
                                     np.array(kernel_size, dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    pfc.append(pf_kernel_size)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    pf_dilation = trt.PluginField('dilation',
                                  np.array(dilation, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_dilation)

    if isinstance(padding, int):
        padding = (padding, padding)
    pf_padding = trt.PluginField('padding', np.array(padding, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    pfc.append(pf_padding)

    if isinstance(stride, int):
        stride = (stride, stride)
    pf_stride = trt.PluginField('stride', np.array(stride, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    return creator.create_plugin(layer_name, pfc)
