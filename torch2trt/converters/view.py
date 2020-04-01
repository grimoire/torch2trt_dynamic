from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.flatten')
@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.view_as')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.unsqueeze')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.squeeze')
def convert_view(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape
        
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    layer = ctx.network.add_shuffle(input_trt)
    if support_dynamic_shape:
        layer.reshape_dims = output.shape
    else:
        layer.reshape_dims = tuple(output.shape[1:])
    output._trt = layer.get_output(0)



class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return View(1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return View(1, 1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_3d():
    return View(1, 1, 1, -1)
