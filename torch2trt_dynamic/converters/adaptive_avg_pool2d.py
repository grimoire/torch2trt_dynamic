import tensorrt as trt
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
def convert_adaptive_avg_pool2d(ctx):
    input = ctx.method_args[0]
    output_size = get_arg(ctx, 'output_size', pos=1, default=0)
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    output_size = tuple([-1 if not o else o for o in output_size])

    if output_size[0] == 1 and output_size[1] == 1:
        # use reduce as max pool2d
        shape_length = len(input.shape)
        axes = (1 << (shape_length - 1)) + (1 << (shape_length - 2))
        keepdim = True
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG,
                                       axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        from torch2trt_dynamic.plugins import create_adaptivepool_plugin
        plugin = create_adaptivepool_plugin(
            'adaptive_avg_pool2d_' + str(id(input)),
            output_size=output_size,
            pooling_type=trt.PoolingType.AVERAGE)

        layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

        output._trt = layer.get_output(0)
