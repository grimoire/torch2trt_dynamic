from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def convert_compare(ctx, compare_op):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, compare_op)
    layer.set_output_type(0, trt.bool)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.gt')
@tensorrt_converter('torch.Tensor.__gt__')
def convert_greater(ctx):
    convert_compare(ctx, trt.ElementWiseOperation.GREATER)


@tensorrt_converter('torch.lt')
@tensorrt_converter('torch.Tensor.__lt__')
def convert_less(ctx):
    convert_compare(ctx, trt.ElementWiseOperation.LESS)


@tensorrt_converter('torch.Tensor.__and__')
def convert_and(ctx):
    convert_compare(ctx, trt.ElementWiseOperation.AND)


@tensorrt_converter('torch.Tensor.__or__')
def convert_or(ctx):
    convert_compare(ctx, trt.ElementWiseOperation.OR)


@tensorrt_converter('torch.eq')
@tensorrt_converter('torch.Tensor.__eq__')
def convert_equal(ctx):
    convert_compare(ctx, trt.ElementWiseOperation.EQUAL)


@tensorrt_converter('torch.ge')
@tensorrt_converter('torch.Tensor.__ge__')
def convert_greaterequal(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    greater = input_a>input_b
    equal = input_a==input_b

    ctx.method_return = greater
    convert_greater(ctx)

    ctx.method_return = equal
    convert_equal(ctx)

    ctx.method_args = [greater, equal]
    ctx.method_return = output
    convert_or(ctx)


@tensorrt_converter('torch.le')
@tensorrt_converter('torch.Tensor.__le__')
def convert_lessequal(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    less = input_a<input_b
    equal = input_a==input_b

    ctx.method_return = less
    convert_less(ctx)

    ctx.method_return = equal
    convert_equal(ctx)

    ctx.method_args = [less, equal]
    ctx.method_return = output
    convert_or(ctx)
