from ..torch2trt_dynamic import *
from ..module_test import add_module_test
import torch


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    ### reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = (0,)*len(input_trt.shape) + (1, 1) 

    ### add fully connected
    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()
    
    layer = ctx.network.add_convolution(
        input=layer.get_output(0),
        num_output_maps=module.out_features,
        kernel_shape=(1, 1),
        kernel=module.weight.detach().cpu().numpy(),
        bias=bias)

    # layer = ctx.network.add_fully_connected(
    #     input=layer.get_output(0),
    #     # input=input_trt,
    #     num_outputs=module.out_features,
    #     kernel=module.weight.detach().cpu().numpy(),
    #     bias=bias)

    ### reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    # # layer.reshape_dims = tuple(output.shape[1:])
    layer.reshape_dims = (0,)*len(input_trt.shape)

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
