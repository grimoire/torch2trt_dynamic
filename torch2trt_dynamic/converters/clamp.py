from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from .size import IntWarper


def __add_clamp(network, trt_input, val, op):
    
    # create TensorRT constant for minimum value
    val_shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    if isinstance(val, IntWarper):
        val_trt = val._trt

        # convert type
        layer = network.add_identity(val_trt)
        layer.set_output_type(0, trt_input.dtype)
        val_trt = layer.get_output(0)

        # convert 2 / to prevent warning, might remove in future version
        layer = network.add_elementwise(val_trt, trt_(network, torch.zeros((1,), dtype=torch.float32)), trt.ElementWiseOperation.SUM)
        layer.set_output_type(0, trt_input.dtype)
        val_trt = layer.get_output(0)

        # reshape
        layer = network.add_shuffle(val_trt)
        layer.reshape_dims = val_shape
        val_trt = layer.get_output(0)
    else:
        # val_shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
        val_tensor = val * torch.ones(val_shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
        layer = network.add_constant(val_shape, val_tensor)
        val_trt = layer.get_output(0)
    layer = network.add_elementwise(trt_input, val_trt, op)
    
    return layer

    
# CLAMP_MIN

    
@tensorrt_converter('torch.clamp_min')
@tensorrt_converter('torch.Tensor.clamp_min')
def convert_clamp_min(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    val = get_arg(ctx, 'min', pos=1, default=0)
    # val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MAX)
    
    output._trt = layer.get_output(0)

    
class TorchClampMin(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_min(x, -0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_min():
    return TorchClampMin()


class TensorClampMin(torch.nn.Module):
    def forward(self, x):
        return x.clamp_min(-0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_min():
    return TensorClampMin()

    
# CLAMP_MAX


@tensorrt_converter('torch.clamp_max')
@tensorrt_converter('torch.Tensor.clamp_max')
def convert_clamp_max(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    val = get_arg(ctx, 'max', pos=1, default=0)
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class TorchClampMax(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_max(x, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_max():
    return TorchClampMax()


class TensorClampMax(torch.nn.Module):
    def forward(self, x):
        return x.clamp_max(0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_max():
    return TensorClampMax()


# CLAMP

    
@tensorrt_converter('torch.clamp')
@tensorrt_converter('torch.Tensor.clamp')
def convert_clamp(ctx):
    input = ctx.method_args[0]
    min_val = get_arg(ctx, 'min', pos=1, default=None)
    max_val = get_arg(ctx, 'max', pos=2, default=None)
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    if min_val is not None:
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
        input_trt = layer.get_output(0)
    if max_val is not None:
        layer = __add_clamp(ctx.network, input_trt, max_val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class TorchClamp(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, -0.1, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp():
    return TorchClamp()


class TensorClamp(torch.nn.Module):
    def forward(self, x):
        return x.clamp(-0.1, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp():
    return TensorClamp()