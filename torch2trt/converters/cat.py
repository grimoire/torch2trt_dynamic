from torch2trt.torch2trt import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape

    inputs = ctx.method_args[0]

    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    else:
        dim = ctx.method_args[1]

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)

    if support_dynamic_shape:
        layer.axis = dim
    else:
        layer.axis = dim - 1
    output._trt = layer.get_output(0)


    
@tensorrt_converter('torch.stack')
def convert_cat(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape
        
    inputs = ctx.method_args[0]

    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    else:
        dim = ctx.method_args[1]

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    if dim==-1:
        dim = len(inputs[0].shape)
    shape = inputs[0].shape[:dim] + (1,) + inputs[0].shape[dim:]
    shape = tuple(shape)
    reshaped_trt_inputs = []
    for trt_input in trt_inputs:
        layer = ctx.network.add_shuffle(trt_input)
        layer.reshape_dims = shape
        reshaped_trt_inputs.append(layer.get_output(0))

    layer = ctx.network.add_concatenation(inputs=reshaped_trt_inputs)

    if support_dynamic_shape:
        layer.axis = dim
    else:
        layer.axis = dim - 1
    output._trt = layer.get_output(0)