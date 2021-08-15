from torch2trt_dynamic.plugins import create_torchunfold_plugin
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.nn.functional.unfold')
def convert_unfold(ctx):
    input = ctx.method_args[0]
    kernel_size = get_arg(ctx, 'kernel_size', pos=1, default=0)
    dilation = get_arg(ctx, 'dilation', pos=2, default=1)
    padding = get_arg(ctx, 'padding', pos=3, default=0)
    stride = get_arg(ctx, 'stride', pos=4, default=1)
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)

    plugin = create_torchunfold_plugin(
        'unfold_' + str(id(input)),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride)

    layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output._trt = layer.get_output(0)
