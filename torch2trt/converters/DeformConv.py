from torch2trt.torch2trt import *

from torch2trt.plugins import *
import mmdet


@tensorrt_converter('mmdet.ops.DeformConv.forward')
def convert_DeformConv(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    offset = ctx.method_args[2]

    input_trt = trt_(ctx.network, input)
    offset_trt = trt_(ctx.network, offset)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    deformable_group = module.deformable_groups
    group = module.groups

    kernel = module.weight.detach().cpu().numpy()

    plugin = create_dcn_plugin("dcn_" + str(id(module)),
                               out_channels=module.out_channels,
                               kernel_size=kernel_size,
                               W=kernel,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               deformable_group=deformable_group,
                               group=group
                               )
                               
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, offset_trt], plugin=plugin)
    
    output._trt = custom_layer.get_output(0)
