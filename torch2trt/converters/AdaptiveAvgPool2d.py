from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .adaptive_avg_pool2d import convert_adaptive_avg_pool2d



@tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert_AdaptiveAvgPool2d(ctx):
    ctx.method_args = (ctx.method_args[1], ctx.method_args[0].output_size)
    convert_adaptive_avg_pool2d(ctx)

### old
# @tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
# def convert_AdaptiveAvgPool2d(ctx):
#     module = ctx.method_args[0]
#     input = ctx.method_args[1]
#     output = ctx.method_return
    
#     input_trt = trt_(ctx.network, input)

#     output_size = module.output_size
#     if not isinstance(output_size, tuple):
#         output_size = (output_size, ) * 2

#     if output_size[0]==1 and output_size[1] == 1:
#         shape_length = len(input.shape)
#         axes = (1<<(shape_length-1)) + (1<<(shape_length-2))
#         keepdim = True
#         layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keepdim)
#         output._trt = layer.get_output(0)
#     else:
#         stride = (input_trt.shape[-2] // output_size[-2], input_trt.shape[-1] // output_size[-1])

#         kernel_size = stride
#         layer = ctx.network.add_pooling(
#             input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
#         layer.stride = stride

#         output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_1x1():
    return torch.nn.AdaptiveAvgPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_2x2():
    return torch.nn.AdaptiveAvgPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_3x3():
    return torch.nn.AdaptiveAvgPool2d((3, 3))
