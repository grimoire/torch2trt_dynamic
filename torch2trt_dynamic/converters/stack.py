from torch2trt_dynamic.torch2trt_dynamic import *
from .cat import *
from .unsqueeze import *


@tensorrt_converter('torch.stack')
def convert_stack(ctx):
    inputs = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    output = ctx.method_return

    unsqueeze_inputs = []
    for input in inputs:
        unsqueeze_input = input.unsqueeze(dim=dim)
        ctx.method_args = (input, dim)
        ctx.method_return = unsqueeze_input
        convert_unsqueeze(ctx)
        unsqueeze_inputs.append(unsqueeze_input)
    
    ctx.method_args = (unsqueeze_inputs, dim)
    ctx.method_return = output

    convert_cat(ctx)


# @tensorrt_converter('torch.stack')
# def convert_stack(ctx):
#     support_dynamic_shape = False
#     if hasattr(ctx, "support_dynamic_shape"):
#         support_dynamic_shape = ctx.support_dynamic_shape
        
#     inputs = ctx.method_args[0]

#     if 'dim' in ctx.method_kwargs:
#         dim = ctx.method_kwargs['dim']
#     else:
#         dim = ctx.method_args[1]

#     output = ctx.method_return
#     trt_inputs = [trt_(ctx.network, i) for i in inputs]

#     if dim==-1:
#         dim = len(inputs[0].shape)
#     shape = inputs[0].shape[:dim] + (1,) + inputs[0].shape[dim:]
#     shape = tuple(shape)
#     reshaped_trt_inputs = []
#     for trt_input in trt_inputs:
#         layer = ctx.network.add_shuffle(trt_input)
#         layer.reshape_dims = shape
#         reshaped_trt_inputs.append(layer.get_output(0))

#     layer = ctx.network.add_concatenation(inputs=reshaped_trt_inputs)

#     if support_dynamic_shape:
#         layer.axis = dim
#     else:
#         layer.axis = dim - 1
#     output._trt = layer.get_output(0)