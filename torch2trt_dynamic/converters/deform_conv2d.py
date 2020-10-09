from ..torch2trt_dynamic import *
from .Conv2d import convert_Conv2d

from ..plugins import *
import torchvision.ops

@tensorrt_converter('torchvision.ops.deform_conv.deform_conv2d')
def convert_deform_conv2d(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    offset = get_arg(ctx, 'offset', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    stride = get_arg(ctx, 'stride', pos=4, default=1)
    padding = get_arg(ctx, 'padding', pos=5, default=0)
    dilation = get_arg(ctx, 'dilation', pos=6, default=1)
    # groups = get_arg(ctx, 'groups', pos=6, default=1)
    # deform_groups = get_arg(ctx, 'deform_groups', pos=7, default=1)
    groups=1

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    offset_trt = trt_(ctx.network, offset)

    kernel_size = weight.shape[2]
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    deform_groups=int(offset.shape[1]//(2*kernel_size[0]*kernel_size[1]))

    kernel = weight.detach().cpu().numpy()
    out_channels = output.shape[1]
    
    bias = bias.detach().cpu().numpy()

    plugin = create_dcn_plugin("dcn_" + str(id(input)),
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               W=kernel,
                               B=bias,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               deformable_group=deform_groups,
                               group=groups
                               )
                               
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, offset_trt], plugin=plugin)
    
    output._trt = custom_layer.get_output(0)
