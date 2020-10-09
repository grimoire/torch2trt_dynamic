from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.plugins import *
import torchvision.ops


@tensorrt_converter('torchvision.ops.nms')
def convert_nms(ctx):

    boxes = get_arg(ctx, 'boxes', pos=0, default=None)
    scores = get_arg(ctx, 'scores', pos=1, default=None)
    iou_threshold = get_arg(ctx, 'iou_threshold', pos=2, default=0.7)

    output = ctx.method_return

    boxes_trt = trt_(ctx.network, boxes)
    scores_trt = trt_(ctx.network, scores)


    plugin = create_nms_plugin("nms_" + str(id(boxes)),
                               iou_threshold=iou_threshold
                               )

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[boxes_trt, scores_trt], plugin=plugin)
    
    output._trt = custom_layer.get_output(0)