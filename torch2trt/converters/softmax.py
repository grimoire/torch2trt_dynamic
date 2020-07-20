from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.softmax')
@tensorrt_converter('torch.softmax')
@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape

    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # get dims from args or kwargs
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    if dim is None:
        dim = -1
    if dim<0:
        dim = len(input.shape)+dim

    # axes = 1 << (dim - 1)
    if not support_dynamic_shape:
        dim -= 1
        if dim<0:
            print("can't do log softmax on batch dims.")
    axes = 1<<dim

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module():
    return torch.nn.Softmax(1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim2():
    return torch.nn.Softmax(2)
