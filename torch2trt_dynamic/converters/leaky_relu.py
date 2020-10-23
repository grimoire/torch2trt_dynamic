from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.leaky_relu')
@tensorrt_converter('torch.nn.functional.leaky_relu_')
def convert_leaky_relu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    negative_slope = get_arg(ctx, 'weight', pos=1, default=None)
    output = ctx.method_return

    input_trt, weight_trt = trt_(ctx.network, input, negative_slope)

    # y = prelu(x) = relu(x) - alpha * relu(-x)
    # weight_shape = [1] * len(input.shape)
    # weight_shape[1] = weight.numel()
    # weight_trt = ctx.network.add_constant(weight_shape, -weight.detach().view(weight_shape).cpu().numpy()).get_output(0) # detach so considered leaf

    # x >= 0
    a = ctx.network.add_activation(input_trt,
                                   trt.ActivationType.RELU).get_output(0)

    # x <= 0
    b = ctx.network.add_unary(input_trt, trt.UnaryOperation.NEG).get_output(0)
    b = ctx.network.add_activation(b, trt.ActivationType.RELU).get_output(0)
    b = ctx.network.add_elementwise(
        b, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)

    # y = a + b
    y = ctx.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)

    output._trt = y.get_output(0)