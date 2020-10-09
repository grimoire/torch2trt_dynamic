from torch2trt_dynamic.torch2trt_dynamic import *

from torch2trt_dynamic.plugins import *

@tensorrt_converter('torch.Tensor.repeat')
def convert_repeat(ctx):
    input = ctx.method_args[0]
    shape = ctx.method_args[1]
    if isinstance(shape, int):
        shape = ctx.method_args[1:]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    plugin = create_repeat_plugin("repeat_" + str(id(input)),
                                  repeat_shape=shape
                                  )
    
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)

