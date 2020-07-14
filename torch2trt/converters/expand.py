import tensorrt as trt
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .repeat import *


@tensorrt_converter('torch.Tensor.expand')
def convert_argmax(ctx):
    
    old_args = ctx.method_args
    input = ctx.method_args[0]
    if isinstance(ctx.method_args[1:], int):
        sizes = ctx.method_args[1:]
    else:
        sizes = ctx.method_args[1]

    output = ctx.method_return

    repeat_shape = []
    for i in range(len(input.shape)):
        repeat_shape.append(output.shape[i]//input.shape[i])
    
    ctx.method_args = [input]+repeat_shape
    convert_repeat(ctx)
    ctx.method_args=old_args
