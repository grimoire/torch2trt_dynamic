# dummy converters throw warnings method encountered

from .dummy_converters import *

# supported converters will override dummy converters

from .activation import *
from .add import *
from .avg_pool2d import *
from .mul import *
from .div import *
from .BatchNorm1d import *
from .BatchNorm2d import *
from .cat import *
from .clamp import *
from .Conv1d import *
from .Conv2d import *
from .ConvTranspose2d import *
from .getitem import *
from .identity import *
from .Identity import *
from .instance_norm import *
from .Linear import *
from .LogSoftmax import *
from .max_pool2d import *
from .max import *
from .min import *
from .normalize import *
from .pad import *
from .permute import *
from .pow import *
from .prelu import *
from .prod import *
from .relu import *
from .ReLU import *
from .relu6 import *
from .ReLU6 import *
from .sigmoid import *
from .sub import *
from .sum import *
from .view import *
from .tanh import *
from .transpose import *
from .mean import *
from .softmax import *
from .split import *
from .chunk import *
from .unary import *

# without plugin
from .matmul import *
from .interpolate_custom import *
from .topk import *
from .index_select import *
from .addcmul import *
from .conv2d import *
from .view_as import *
from .unsqueeze import *
from .squeeze import *
from .flatten import *
from .stack import *
from .pixel_shuffle import *
from .LayerNorm import *
from .exview import *
from .size import *
from .argmax import *
from .argmin import *
from .cast_type import *
from .logical import *
from .std import *
from .masked_fill import *
from .mod import convert_mod
from .narrow import convert_narrow
from .ConvTranspose1d import convert_ConvTranspose1d
from .zeros_like import convert_zeros_like
from .ones_like import convert_ones_like
from .full_like import convert_full_like
from .new_zeros import convert_new_zeros
from .new_ones import convert_new_ones
from .arange import convert_arange
from .linspace import convert_linspace
from .to import convert_Tensor_to
from .floor_divide import convert_floor_div, convert_rfloor_div
from .zeros import convert_zeros
from .ones import convert_ones
from .t import convert_t
from .linear import convert_linear

try:
    # custom plugin support
    from .GroupNorm import *
    from .repeat import *
    from .expand import *
    from .gather import *
    from .adaptive_avg_pool2d import *
    from .adaptive_max_pool2d import *
    from .AdaptiveAvgPool2d import *
    from .AdaptiveMaxPool2d import *
    from .meshgrid import convert_meshgrid
    from .grid_sample import convert_grid_sample
    from .flip import convert_flip
    from .cummax import convert_cummax
    from .cummin import convert_cummin
    from .cumsum import convert_cumsum
    from .cumprod import convert_cumprod
    from .expand_as import convert_expand_as
    from .deform_conv2d import convert_deform_conv2d
    from .nms import convert_nms
    from .roi_align import convert_roi_align, convert_RoiAlign
    from .roi_pool import convert_roi_pool, convert_RoIPool
except:
    print("plugin not found.")
