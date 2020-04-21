from torch2trt.torch2trt import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape

    inputs = ctx.method_args[0]

    dim = get_arg(ctx, 'dim', pos=1, default=0)
    if dim<0:
        dim = len(inputs[0].shape)+dim

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)

    if support_dynamic_shape:
        layer.axis = dim
    else:
        layer.axis = dim - 1
    output._trt = layer.get_output(0)

