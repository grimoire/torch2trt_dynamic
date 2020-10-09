from ..torch2trt_dynamic import *

from ..plugins import *

@tensorrt_converter('torch.Tensor.expand_as')
def convert_expand_as(ctx):
    input = ctx.method_args[0]
    other = get_arg(ctx, 'other', pos=1, default=None)

    input_trt = trt_(ctx.network, input)
    other_trt = trt_(ctx.network, other)
    output = ctx.method_return

    plugin = create_repeat_plugin("repeat_" + str(id(input)),
                                  repeat_shape=[]
                                  )
                                  
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, other_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
    