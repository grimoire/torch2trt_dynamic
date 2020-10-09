import tensorrt as trt
import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))



def create_nms_plugin(layer_name,
                      iou_threshold):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchNMSPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()
    
    pf_iou_threshold = trt.PluginField("iou_threshold", np.array(
        [iou_threshold], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_iou_threshold)

    return creator.create_plugin(layer_name, pfc)