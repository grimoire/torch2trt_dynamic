import tensorrt as trt
from torch2trt_dynamic.plugins import create_adaptivepool_plugin
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.nn.functional.adaptive_avg_pool1d')
def convert_adaptive_avg_pool1d(ctx):
    input = ctx.method_args[0]
    output_size = get_arg(ctx, 'output_size', pos=1, default=0)
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)

    if output_size == 1:
        # use reduce as max pool2d
        shape_length = len(input.shape)
        axes = (1 << (shape_length - 1))
        keepdim = True
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG,
                                       axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        output_size = (output_size, 1)

        # input.unsqueeze(-1)
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (0, 0, 0, 1)
        input_trt = layer.get_output(0)

        # adaptive pool 2d
        plugin = create_adaptivepool_plugin(
            'adaptive_avg_pool2d_' + str(id(input)),
            output_size=output_size,
            pooling_type=trt.PoolingType.AVERAGE)

        layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

        output_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = (0, 0, 0)
        output_trt = layer.get_output(0)

        output._trt = output_trt
