from torch2trt.torch2trt import *

from torch2trt.plugins import *
import mmdet


@tensorrt_converter('mmdet.ops.carafe.CARAFEPack.kernel_normalizer')
def convert_Repeat(ctx):
    module = ctx.method_args[0]
    mask = ctx.method_args[1]
    
    scale_factor = module.scale_factor
    up_kernel = module.up_kernel

    mask_trt = trt_(ctx.network, mask)
    output = ctx.method_return

    plugin = create_carafekernelnormalize_plugin("carafekernelnorm_" + str(id(module)),
                                       scale_factor,
                                       up_kernel)
                                       
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[mask_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
