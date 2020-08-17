import tensorrt as trt
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
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


    size_diff = len(output.shape) - len(input.shape)
    if size_diff>0:
        view_input = input.view(1,1,*input.shape)
        ctx.method_args = [input,[input],["1"]*size_diff + ["a{}".format(idx) for idx in range(input.dim())]]
        ctx.method_return = view_input
        convert_exview(ctx)
        input = view_input

    repeat_shape = []
    for i in range(len(input.shape)):
        repeat_shape.append(output.shape[i]//input.shape[i])
    
    ctx.method_args = [input]+repeat_shape
    ctx.method_return = output
    convert_repeat(ctx)
    ctx.method_args=old_args
