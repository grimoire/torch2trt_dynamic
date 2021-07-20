from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.Tensor.view_as')
def convert_view_as(ctx):

    input = ctx.method_args[0]
    other = get_arg(ctx, 'other', pos=1, default=None)
    input_trt = trt_(ctx.network, input)
    other_trt = trt_(ctx.network, other)
    output = ctx.method_return

    shape_trt = ctx.network.add_shape(other_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, shape_trt)
    output._trt = layer.get_output(0)
