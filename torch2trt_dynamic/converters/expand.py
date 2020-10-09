import tensorrt as trt
from ..torch2trt_dynamic import *
from ..module_test import add_module_test
from .repeat import *
from .exview import convert_exview


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    
    old_args = ctx.method_args
    input = ctx.method_args[0]
    if isinstance(ctx.method_args[1:], int):
        sizes = ctx.method_args[1:]
    else:
        sizes = ctx.method_args[1]

    output = ctx.method_return

    repeat_shape = []
    for i in range(output.dim()):
        if i < output.dim()-input.dim():
            repeat_shape.append(output.shape[i])
        else:
            repeat_shape.append(output.shape[i]//input.shape[i+input.dim()-output.dim()])
    
    ctx.method_args = [input]+repeat_shape
    ctx.method_return = output
    convert_repeat(ctx)
    ctx.method_args=old_args
