from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from .unary import UnaryModule
from .flatten import *
from .topk import *
from .squeeze import *


def __convert_min_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MIN)
    output._trt = layer.get_output(0)
    

def __convert_min_reduce(ctx):

    if isinstance(ctx.method_return, torch.Tensor):
        input = ctx.method_args[0]
        dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(0, input.ndim)))
        keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
        input_trt= trt_(ctx.network, input)
        output_val = ctx.method_return
        layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.MIN, torch_dim_to_trt_axes(dim), keepdim)
        output_val._trt = layer.get_output(0)
        return

    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    input = ctx.method_args[0]
    output = ctx.method_return

    dim = get_arg(ctx, 'dim', pos=1, default=None)
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)

    return_single = False

    # dim is None
    if dim is None:
        return_single = True
        input_flatten = input.flatten()
        ctx.method_args = [input]
        ctx.method_return = input_flatten
        convert_flatten(ctx)
        input = ctx.method_return
        dim = 0
    
    # topk
    topk_output = input.topk(1, dim, False)
    topk_input = [input, 1, dim, False]
    ctx.method_args = topk_input
    ctx.method_kwargs = {}
    ctx.method_return = topk_output
    convert_topk(ctx)
    topk_value = ctx.method_return[0]
    topk_index = ctx.method_return[1]


    # keepdim
    if not keepdim and topk_index.shape[dim]==1 and len(topk_index.shape)>1:

        topk_index_squeeze = topk_index.squeeze(dim)
        ctx.method_args = [topk_index, dim]
        ctx.method_return = topk_index_squeeze
        convert_squeeze(ctx)

        topk_value_squeeze = topk_value.squeeze(dim)
        ctx.method_args = [topk_value, dim]
        ctx.method_return = topk_value_squeeze
        convert_squeeze(ctx)

        topk_index = topk_index_squeeze
        topk_value = topk_value_squeeze

    if return_single:
        output._trt = topk_value._trt
    else:
        output[0]._trt = topk_value._trt
        output[1]._trt = topk_index._trt

    ctx.method_return = output

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs

    

@tensorrt_converter('torch.min')
@tensorrt_converter('torch.Tensor.min')
def convert_min(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_min_elementwise(ctx)
    else:
        __convert_min_reduce(ctx)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1():
    return UnaryModule(lambda x: torch.min(x, 1)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim22():
    return UnaryModule(lambda x: torch.min(x, 2)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.min(x, 1, keepdim=True)[0])


class MinElementwise(torch.nn.Module):
    def forward(self, x, y):
        return torch.min(x, y)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
def test_min_elementwise():
    return MinElementwise()
