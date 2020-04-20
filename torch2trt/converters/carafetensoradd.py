from torch2trt.torch2trt import *

from torch2trt.plugins import *
import mmdet


@tensorrt_converter('mmdet.models.necks.fpn_carafe.FPN_CARAFE.tensor_add')
def convert_Repeat(ctx):
    module = ctx.method_args[0]
    x = ctx.method_args[1]
    mask = ctx.method_args[2]

    x_trt = trt_(ctx.network, x)
    mask_trt = trt_(ctx.network, mask)
    output = ctx.method_return

    plugin = create_carafetensoradd_plugin("carafetensoradd_" + str(id(module)),)
                                       
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[x_trt, mask_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
