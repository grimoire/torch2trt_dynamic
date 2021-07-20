from torch2trt_dynamic.plugins import create_meshgrid_plugin
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.meshgrid')
def convert_meshgrid(ctx):
    input_list = ctx.method_args
    output = ctx.method_return

    input_list_trt = [
        trt_(ctx.network, input_tensor) for input_tensor in input_list
    ]

    num_inputs = len(input_list)

    plugin = create_meshgrid_plugin(
        'adaptive_meshgrid_' + str(id(input)), num_inputs=num_inputs)

    layer = ctx.network.add_plugin_v2(inputs=input_list_trt, plugin=plugin)

    for idx, out in enumerate(output):
        out._trt = layer.get_output(idx)
