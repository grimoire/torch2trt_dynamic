from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.take')
def convert_take(ctx):
    input = ctx.method_args[0]
    index = get_arg(ctx, 'index', pos=1, default=None)

    input_trt = trt_(ctx.network, input)
    index_trt = trt_(ctx.network, index)
    output = ctx.method_return

    # flatten input
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = (-1, )
    flatten_input_trt = layer.get_output(0)

    # flatten index
    output_trt = ctx.network.add_gather(flatten_input_trt, index_trt,
                                        0).get_output(0)

    output._trt = output_trt
