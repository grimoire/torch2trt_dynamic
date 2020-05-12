from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.adaptive_max_pool2d')
def convert_adaptive_max_pool2d(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)

    output_size = ctx.method_args[1]
    if isinstance(output_size, int):
        output_size = (output_size, ) * 2

    if output_size[0]==1 and output_size[1] == 1:
        shape_length = len(input.shape)
        axes = (1<<(shape_length-1)) + (1<<(shape_length-2))
        keepdim = True
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.MAX, axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        stride = (input._trt.shape[-2] // output_size[-2], input._trt.shape[-1] // output_size[-1])

        kernel_size = stride
        layer = ctx.network.add_pooling(
            input=input._trt, type=trt.PoolingType.MAX, window_size=kernel_size)
        layer.stride = stride

        output._trt = layer.get_output(0)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_max_pool2d_1x1():
    return torch.nn.AdaptiveMaxPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_max_pool2d_2x2():
    return torch.nn.AdaptiveMaxPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_max_pool2d_3x3():
    return torch.nn.AdaptiveMaxPool2d((3, 3))
