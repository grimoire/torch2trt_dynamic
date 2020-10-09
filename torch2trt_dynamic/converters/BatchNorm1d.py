from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.nn.BatchNorm1d.forward')
def convert_BatchNorm1d(ctx):

    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    # reshape to 2D
    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    one_trt = trt_(ctx.network, torch.tensor([1],dtype=torch.int32).to(input.device))
    if len(input.shape)==2:
        new_input_shape_trt = ctx.network.add_concatenation([input_shape_trt, one_trt, one_trt]).get_output(0)
    else:
        new_input_shape_trt = ctx.network.add_concatenation([input_shape_trt, one_trt]).get_output(0)
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_input_shape_trt)

    layer = ctx.network.add_scale(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)

    # reshape back to 1D
    conv_out_trt = layer.get_output(0)
    layer = ctx.network.add_shuffle(conv_out_trt)
    layer.set_input(1, input_shape_trt)
    
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_BatchNorm1d_basic():
    return torch.nn.BatchNorm1d(10)