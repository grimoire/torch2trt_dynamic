from torch2trt_dynamic.torch2trt_dynamic import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    inputs = ctx.method_args[0]

    dim = get_arg(ctx, 'dim', pos=1, default=0)
    if dim<0:
        dim = len(inputs[0].shape)+dim

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)

    layer.axis = dim
    output._trt = layer.get_output(0)

