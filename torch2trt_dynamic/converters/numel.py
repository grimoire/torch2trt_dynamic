import torch
from torch2trt_dynamic.torch2trt_dynamic import *
from .size import IntWarper

    
@tensorrt_converter('torch.Tensor.numel')
def convert_numel(ctx):
    input = ctx.method_args[0]

    input_trt = trt_(ctx.network, input)
    shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
    num = ctx.method_return
    
    num_trt = slice_shape_trt(ctx.network, shape_trt, 0, 1)
    for i in range(1, len(input.shape)):
        other_trt = slice_shape_trt(ctx.network, shape_trt, i, 1)
        num_trt = ctx.network.add_elementwise(num_trt, other_trt, trt.ElementWiseOperation.PROD).get_output(0)
    intwarper = IntWarper(num)
    intwarper._trt = num_trt

    ctx.method_return = intwarper
