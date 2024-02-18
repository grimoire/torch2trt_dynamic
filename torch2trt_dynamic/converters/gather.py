import tensorrt as trt

from ..torch2trt_dynamic import get_arg, tensorrt_converter, trt_


@tensorrt_converter('torch.Tensor.gather')
@tensorrt_converter('torch.gather')
def convert_gather(ctx):
    inputs = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    index = get_arg(ctx, 'index', pos=2, default=None)
    output = ctx.method_return

    inputs_trt = trt_(ctx.network, inputs)
    index_trt = trt_(ctx.network, index)

    layer = ctx.network.add_gather_v2(inputs_trt, index_trt,
                                      trt.GatherMode.ELEMENT)
    layer.num_elementwise_dims = 0
    layer.axis = dim

    output._trt = layer.get_output(0)
