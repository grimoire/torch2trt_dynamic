from ..plugins import *
from ..torch2trt_dynamic import *
from .cast_type import convert_type


@tensorrt_converter('torch.cumsum')
@tensorrt_converter('torch.Tensor.cumsum')
def convert_cumsum(ctx):
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    cum_type = 0

    if dim < 0:
        dim = len(input.shape) + dim
    output = ctx.method_return

    if input.dtype == torch.bool or input.dtype == bool:
        cast_input = input.type_as(output)
        ctx.method_args = [input]
        ctx.method_kwargs = {}
        ctx.method_return = cast_input
        convert_type(ctx, torch_dtype_to_trt(output.dtype))
        input_trt = trt_(ctx.network, cast_input)
    else:
        input_trt = trt_(ctx.network, input)

    plugin = create_torchcum_plugin('cumsum_' + str(id(input)),
                                    dim=dim,
                                    cum_type=cum_type)

    custom_layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output_trt = custom_layer.get_output(0)

    if input.dtype != output.dtype:
        tmp_output = output.clone()
        tmp_output._trt = output_trt
        ctx.method_args = [tmp_output]
        ctx.method_kwargs = {}
        ctx.method_return = output
        convert_type(ctx, torch_dtype_to_trt(output.dtype))
        output_trt = ctx.method_return._trt

    output._trt = output_trt

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
    ctx.method_return = output
