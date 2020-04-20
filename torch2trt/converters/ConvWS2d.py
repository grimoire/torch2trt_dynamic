from .Conv2d import *


module_path = 'mmdet.models.utils.conv_ws.ConvWS2d.forward'
try:
    from mmdet.models.utils.conv_ws import ConvWS2d
except:
    module_path = "mmdet.ops.conv_ws.ConvWS2d.forward"
    try:
        from mmdet.ops.conv_ws import ConvWS2d
    except:
        print(module_path, "not found")

@tensorrt_converter(module_path)
def convert_ConvWS2d(ctx):
    convert_Conv2d(ctx)