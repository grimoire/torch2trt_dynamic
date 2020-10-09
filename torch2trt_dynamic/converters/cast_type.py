import tensorrt as trt
from torch2trt_dynamic.torch2trt_dynamic import *


def convert_type(ctx, data_type):
    input = ctx.method_args[0]
    output = ctx.method_return

    input_trt = trt_(ctx.network, input)

    layer = ctx.network.add_identity(input_trt)
    layer.set_output_type(0, data_type)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.long')
@tensorrt_converter('torch.Tensor.int')
def convert_int(ctx):
    convert_type(ctx, trt.DataType.INT32)
    convert_type(ctx, trt.DataType.INT32)

@tensorrt_converter('torch.Tensor.float')
def convert_float(ctx):
    convert_type(ctx, trt.DataType.FLOAT)
    convert_type(ctx, trt.DataType.FLOAT)

# @tensorrt_converter('torch.Tensor.char')
# def convert_char(ctx):
#     convert_type(ctx, trt.DataType.CHAR)


# @tensorrt_converter('torch.Tensor.half')
# def convert_half(ctx):
#     convert_type(ctx, trt.DataType.HALF)


@tensorrt_converter('torch.Tensor.bool')
def convert_bool(ctx):
    convert_type(ctx, trt.DataType.BOOL)



@tensorrt_converter('torch.Tensor.type_as')
def convert_type_as(ctx):
    input = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    other_trt = trt_(ctx.network, other)

    layer = ctx.network.add_identity(input_trt)
    layer.set_output_type(0, other_trt.dtype)
    output._trt = layer.get_output(0)