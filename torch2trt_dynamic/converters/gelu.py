import math

import tensorrt as trt
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.nn.functional.gelu')
def convert_gelu(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    input_trt, b_trt, v1_trt, v0_5_trt, v3_trt = trt_(ctx.network, input,
                                                      0.044715, 1, 0.5, 3)

    layer = ctx.network.add_elementwise(input_trt, v3_trt,
                                        trt.ElementWiseOperation.POW)
    input_p3_trt = layer.get_output(0)

    # b*x**3
    layer = ctx.network.add_elementwise(input_p3_trt, b_trt,
                                        trt.ElementWiseOperation.PROD)
    bx3_trt = layer.get_output(0)

    # x + b*x**3
    layer = ctx.network.add_elementwise(bx3_trt, input_trt,
                                        trt.ElementWiseOperation.SUM)
    xabx3_trt = layer.get_output(0)

    # tanh()
    layer = ctx.network.add_activation(xabx3_trt,
                                       trt.ActivationType.SCALED_TANH)
    layer.alpha = 1
    layer.beta = math.sqrt(2 / math.pi)
    tanh_trt = layer.get_output(0)

    # 1+tanh()
    layer = ctx.network.add_elementwise(tanh_trt, v1_trt,
                                        trt.ElementWiseOperation.SUM)
    oneatanh_trt = layer.get_output(0)

    # x*()
    layer = ctx.network.add_elementwise(input_trt, oneatanh_trt,
                                        trt.ElementWiseOperation.PROD)
    xtanh_trt = layer.get_output(0)

    # output
    layer = ctx.network.add_elementwise(xtanh_trt, v0_5_trt,
                                        trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)
