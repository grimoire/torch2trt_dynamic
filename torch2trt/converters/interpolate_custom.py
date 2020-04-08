import torch
from torch2trt.torch2trt import *

@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):

    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape
    input = ctx.method_args[0]

    try:
        scale_factor = get_arg(ctx, 'scale_factor', pos=2, default=None)
    except KeyError:
        scale_factor = None

    try:
        size = get_arg(ctx, 'size', pos=1, default=None)
    except KeyError:
        size = None

    try:
        mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    except KeyError:
        mode = 'nearest'

    try:
        align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)
    except KeyError:
        align_corners = False

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    if support_dynamic_shape and isinstance(size, torch.Size):
        if scale_factor is None:
            scale_factor = [1]*len(input.shape)
        for i in range(len(input.shape)):
            scale_factor[i] = output.shape[i]/input.shape[i]

    layer = ctx.network.add_resize(input_trt)
    if scale_factor is not None:
        if isinstance(scale_factor, (float,int)):
            scale_factor = [1,1, scale_factor, scale_factor]
        if not support_dynamic_shape:
            scale_factor = scale_factor[1:]
        layer.scales = scale_factor
    else:
        if support_dynamic_shape:
            layer.shape = tuple(output.shape)
        else:
            layer.shape = tuple(output.shape[1:])
    layer.align_corners = align_corners

    if mode=="nearest":
        layer.resize_mode = trt.ResizeMode.NEAREST
    elif mode=="linear":
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.LINEAR
        print("unknown interpolate type, use linear insteed.")

    output._trt = layer.get_output(0)

    
