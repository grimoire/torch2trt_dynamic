from ..torch2trt_dynamic import *
from ..module_test import add_module_test
import torch
from .t import convert_t
from .matmul import convert_matmul
from .sum import convert_sum


@tensorrt_converter('torch.nn.functional.linear')
def convert_linear(ctx):
    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    output = ctx.method_return

    # transpose weight
    weight_transpose = weight.t()
    ctx.method_args = [weight]
    ctx.method_kwargs = {}
    ctx.method_return = weight_transpose
    convert_t(ctx)

    # matmul
    matmul_output = input.matmul(weight_transpose)
    ctx.method_args = [input, weight]
    ctx.method_kwargs = {}
    ctx.method_return = matmul_output
    convert_matmul(ctx)

    # add bias
    if bias is not None:
        add_bias_output = matmul_output + bias
        ctx.method_args = [matmul_output, bias]
        ctx.method_return = add_bias_output
        convert_sum(ctx)
        output._trt = add_bias_output._trt
    else:
        output._trt = matmul_output._trt


    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output
