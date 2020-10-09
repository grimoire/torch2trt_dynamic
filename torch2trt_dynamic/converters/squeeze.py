from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from .identity import *


@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.squeeze')
def convert_squeeze(ctx):
        
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    if dim is None:
        dim = list(filter(lambda x:input.shape[x]==1, range(len(input.shape))))
    else:
        if input.shape[dim]!=1:
            ctx.method_args = [input]
            convert_identity(ctx)
            return
        if dim <0:
            dim = len(input.shape)+dim
        dim = [dim]
    input_trt = trt_(ctx.network, input)
    shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    output = ctx.method_return

    reverse_dim = list(filter(lambda x: x not in dim, range(len(input.shape))))
    reverse_dim_trt = trt_(ctx.network, torch.tensor(reverse_dim,dtype=torch.int32).to(input.device))

    new_shape_trt = ctx.network.add_gather(shape_trt, reverse_dim_trt, 0).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)