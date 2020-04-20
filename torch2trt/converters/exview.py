from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from torch2trt.plugins import *

module_path1 = 'mmdet.ops.context_block.ex_view'
try:
    from mmdet.ops.context_block import ex_view
except:
    print(module_path1, "not found")

module_path2 = 'mmdet.models.plugins.generalized_attention.ex_view'
try:
    from mmdet.models.plugins.generalized_attention import ex_view
except:
    module_path2 = "mmdet.ops.generalized_attention.ex_view"
    try:
        from mmdet.ops.generalized_attention import ex_view
    except:
        print(module_path2, "not found")

@tensorrt_converter(module_path1)
@tensorrt_converter(module_path2)
def convert_exview(ctx):
    input = ctx.method_args[0]
    tensors = ctx.method_args[1]
    exps = ctx.method_args[2]
    input_trt = trt_(ctx.network, input)
    tensors_trt = [trt_(ctx.network, t) for t in tensors]
    output = ctx.method_return

    plugin = create_exview_plugin("exview_" + str(id(input)), exps)

    layer_input = [input_trt] + tensors_trt
    
    custom_layer = ctx.network.add_plugin_v2(
        inputs=layer_input, plugin=plugin)

    output._trt = custom_layer.get_output(0)
