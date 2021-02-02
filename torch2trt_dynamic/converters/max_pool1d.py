from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from .squeeze import convert_squeeze
from .unsqueeze import convert_unsqueeze


@tensorrt_converter('torch.nn.functional.max_pool1d')
def convert_max_pool1d(ctx):
    # parse args
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    input = get_arg(ctx, 'input', pos=0, default=None)
    kernel_size = get_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = get_arg(ctx, 'stride', pos=2, default=None)
    padding = get_arg(ctx, 'padding', pos=3, default=0)
    dilation = get_arg(ctx, 'dilation', pos=4, default=1)
    ceil_mode = get_arg(ctx, 'ceil_mode', pos=5, default=False)

    kernel_size = (kernel_size, 1)
    stride = (stride, 1)
    padding = (padding, 0)

    output = ctx.method_return

    # unsqueeze -1
    unsqueeze_input = input.unsqueeze(-1)
    ctx.method_args = [input, -1]
    ctx.method_kwargs = {}
    ctx.method_return = unsqueeze_input
    convert_unsqueeze(ctx)

    # pool2d
    input_trt = trt_(ctx.network, unsqueeze_input)

    layer = ctx.network.add_pooling(input=input_trt,
                                    type=trt.PoolingType.MAX,
                                    window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    pool2d_output = torch.nn.functional.max_pool2d(unsqueeze_input,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   ceil_mode=ceil_mode)
    pool2d_output._trt = layer.get_output(0)

    # squeeze -1
    ctx.method_args = [pool2d_output, -1]
    ctx.method_kwargs = {}
    ctx.method_return = output
    convert_squeeze(ctx)

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
    ctx.method_return = output