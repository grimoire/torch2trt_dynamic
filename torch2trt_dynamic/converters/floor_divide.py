from ..torch2trt_dynamic import *
from ..module_test import add_module_test


@tensorrt_converter('torch.floor_divide')
@tensorrt_converter('torch.Tensor.floor_divide')
@tensorrt_converter('torch.Tensor.floor_divide_')
@tensorrt_converter('torch.Tensor.__floordiv__')
@tensorrt_converter('torch.Tensor.__ifloordiv__')
def convert_floor_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rfloordiv__')
def convert_rfloor_div(ctx):
    input_a = ctx.method_args[1]  # inputs switched for rdiv
    input_b = ctx.method_args[0]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    output._trt = layer.get_output(0)
    
