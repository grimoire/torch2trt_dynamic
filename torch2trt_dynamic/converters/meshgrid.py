from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.plugins import create_meshgrid_plugin


@tensorrt_converter("torch.meshgrid")
def convert_meshgrid(ctx):
    input_list = ctx.method_args
    output = ctx.method_return

    input_list_trt = [trt_(ctx.network, input_tensor) for input_tensor in input_list]

    num_inputs = len(input_list)


    plugin = create_meshgrid_plugin("adaptive_meshgrid_"+str(id(input)),
                                        num_inputs=num_inputs)
            
    layer = ctx.network.add_plugin_v2(
        inputs=input_list_trt, plugin=plugin)

    for idx, out in enumerate(output):
        out._trt = layer.get_output(idx)


# from .repeat import convert_repeat
# from .view import convert_view
# @tensorrt_converter('torch.meshgrid')
# def convert_meshgrid(ctx):

#     input_list = ctx.method_args
#     output = ctx.method_return

#     num_inputs = len(input_list)

#     for i in range(num_inputs):
#         tmp_in = input_list[i]
#         tmp_out = output[i]

#         shape = [1]*num_inputs
#         shape[i]=-1
#         tmp_in_view = tmp_in.view(*shape)
#         ctx.method_args = [tmp_in, *shape]
#         ctx.method_return = tmp_in_view
#         convert_view(ctx)

#         repeat_shape = [input.view(-1).shape[0] for input in input_list]
#         repeat_shape[i] = 1
#         ctx.method_args = [tmp_in_view, *repeat_shape]
#         ctx.method_return = tmp_out
#         convert_repeat(ctx)

#     ctx.method_args = input_list
#     ctx.method_return = output


