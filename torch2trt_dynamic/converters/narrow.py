from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.plugins import *
from .size import get_intwarper_trt


@tensorrt_converter('torch.Tensor.narrow')
@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    input = ctx.method_args[0]
    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    elif 'dimension' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dimension']
    else:
        dim = ctx.method_args[1]
    input_dim = input.dim()
    if dim<0:
        dim = dim+input_dim

    start = get_arg(ctx, 'start', pos=2, default=None)
    length = get_arg(ctx, 'length', pos=3, default=None)

    
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)

    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
    start_trt = get_intwarper_trt(start, ctx)
    length_trt = get_intwarper_trt(length, ctx)
    stride_trt = trt_(ctx.network, torch.ones([input_dim]).int())
    if dim!=0:
        start_pre_trt = trt_(ctx.network, torch.zeros([dim,]).int())
        start_trt = ctx.network.add_concatenation([start_pre_trt, start_trt]).get_output(0)
        length_pre_trt = slice_shape_trt(ctx.network, input_shape_trt, 0, dim)
        length_trt = ctx.network.add_concatenation([length_pre_trt, length_trt]).get_output(0)
    if dim<input_dim-1:
        start_post_trt = trt_(ctx.network, torch.zeros([input_dim-dim - 1]).int())

        start_trt = ctx.network.add_concatenation([start_trt, start_post_trt]).get_output(0)
        length_post_trt = slice_shape_trt(ctx.network, input_shape_trt, dim+1)
        length_trt = ctx.network.add_concatenation([length_trt, length_post_trt]).get_output(0)
    
    layer = ctx.network.add_slice(input_trt, [0]*input_dim, [1]*input_dim, [1]*input_dim)
    layer.set_input(1, start_trt)
    layer.set_input(2, length_trt)
    layer.set_input(3, stride_trt)
    output._trt = layer.get_output(0)