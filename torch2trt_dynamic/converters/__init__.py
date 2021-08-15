# dummy converters throw warnings method encountered

try:
    from .dummy_converters import *  # noqa: F401,F403
except Exception:
    print('dummy converters not found.')

# supported converters will override dummy converters

from .activation import (convert_elu, convert_leaky_relu, convert_selu,
                         convert_softplus, convert_softsign)
from .add import (convert_add, test_add_basic, test_add_iadd,
                  test_add_radd_float, test_add_radd_int, test_add_torchadd)
from .addcmul import convert_addcmul, test_addcmul
from .arange import convert_arange
from .argmax import convert_argmax
from .argmin import convert_argmin
from .avg_pool2d import convert_avg_pool2d
from .BatchNorm1d import convert_BatchNorm1d, test_BatchNorm1d_basic
from .BatchNorm2d import convert_BatchNorm2d
from .cast_type import (convert_bool, convert_float, convert_int,
                        convert_type_as)
from .cat import convert_cat
from .chunk import (convert_chunk, test_tensor_chunk_3_2, test_torch_chunk_1_1,
                    test_torch_chunk_2_1, test_torch_chunk_3_1,
                    test_torch_chunk_3_2)
from .clamp import convert_clamp, convert_clamp_max, convert_clamp_min
from .Conv1d import convert_Conv1d
from .Conv2d import convert_Conv2d
from .conv2d import convert_conv2d
from .ConvTranspose1d import convert_ConvTranspose1d
from .ConvTranspose2d import convert_ConvTranspose2d
from .div import convert_div, convert_rdiv
from .exview import convert_exview
from .flatten import convert_flatten
from .flip import convert_flip
from .floor_divide import convert_floor_div, convert_rfloor_div
from .full import convert_full
from .full_like import convert_full_like
from .gelu import convert_gelu
from .getitem import convert_tensor_getitem
from .GRU import convert_GRU
from .identity import convert_identity
from .Identity import convert_Identity
from .index_select import convert_index_select
from .instance_norm import convert_instance_norm
from .interpolate_custom import convert_interpolate
from .LayerNorm import convert_LayerNorm
from .Linear import convert_Linear
from .linear import convert_linear
from .linspace import convert_linspace
from .logical import (convert_and, convert_equal, convert_greater,
                      convert_greaterequal, convert_less, convert_lessequal,
                      convert_ne, convert_or, convert_xor)
from .LogSoftmax import convert_LogSoftmax
from .masked_fill import convert_masked_fill
from .matmul import convert_matmul
from .max import convert_max
from .max_pool1d import convert_max_pool1d
from .max_pool2d import convert_max_pool2d
from .mean import convert_mean
from .meshgrid import convert_meshgrid
from .min import convert_min
from .mod import convert_mod
from .mul import convert_mul
from .narrow import convert_narrow
from .new_ones import convert_new_ones
from .new_zeros import convert_new_zeros
from .normalize import convert_normalize
from .numel import convert_numel
from .ones import convert_ones
from .ones_like import convert_ones_like
from .pad import convert_pad
from .permute import convert_permute
from .pixel_shuffle import convert_pixel_shuffle
from .pow import convert_pow, convert_rpow
from .prelu import convert_prelu
from .prod import convert_prod
from .relu import convert_relu
from .ReLU import convert_ReLU
from .relu6 import convert_relu6
from .ReLU6 import convert_ReLU6
from .repeat import convert_expand, convert_expand_as, convert_repeat
from .roll import convert_roll
from .sigmoid import convert_sigmoid
from .size import (convert_intwarper_add, convert_intwarper_floordiv,
                   convert_intwarper_mul, convert_intwarper_pow,
                   convert_intwarper_radd, convert_intwarper_rfloordiv,
                   convert_intwarper_rmul, convert_intwarper_rpow,
                   convert_intwarper_rsub, convert_intwarper_sub,
                   convert_shapewarper_numel, convert_size)
from .softmax import convert_softmax
from .split import convert_split
from .squeeze import convert_squeeze
from .stack import convert_stack
from .std import convert_std
from .sub import convert_rsub, convert_sub
from .sum import convert_sum
from .t import convert_t
from .take import convert_take
from .tanh import convert_tanh
from .to import convert_Tensor_to
from .topk import convert_topk
from .transpose import convert_transpose
from .unary import (convert_abs, convert_acos, convert_asin, convert_atan,
                    convert_ceil, convert_cos, convert_cosh, convert_floor,
                    convert_invert, convert_log, convert_log2, convert_neg,
                    convert_reciprocal, convert_sin, convert_sinh,
                    convert_sqrt, convert_tan)
from .unsqueeze import convert_unsqueeze
from .view import convert_view
from .view_as import convert_view_as
from .where import convert_Tensor_where, convert_where
from .zeros import convert_zeros
from .zeros_like import convert_zeros_like

__all__ = []
# activation
__all__ += [
    'convert_leaky_relu',
    'convert_elu',
    'convert_selu',
    'convert_softsign',
    'convert_softplus',
]
# add
__all__ += [
    'convert_add', 'test_add_basic', 'test_add_iadd', 'test_add_radd_float',
    'test_add_radd_int', 'test_add_torchadd'
]
# addcmul
__all__ += ['convert_addcmul', 'test_addcmul']
# arange
__all__ += ['convert_arange']
# argmax
__all__ += ['convert_argmax']
# argmin
__all__ += ['convert_argmin']
# avg_pool2d
__all__ += ['convert_avg_pool2d']
# BatchNorm1d
__all__ += ['convert_BatchNorm1d', 'test_BatchNorm1d_basic']
# BatchNorm2d
__all__ += ['convert_BatchNorm2d']
# cast_type
__all__ += ['convert_bool', 'convert_float', 'convert_int', 'convert_type_as']
# cat
__all__ += ['convert_cat']
# chunk
__all__ += [
    'convert_chunk', 'test_torch_chunk_1_1', 'test_torch_chunk_2_1',
    'test_torch_chunk_3_1', 'test_torch_chunk_3_2', 'test_tensor_chunk_3_2'
]
# clamp
__all__ += [
    'convert_clamp',
    'convert_clamp_max',
    'convert_clamp_min',
]
# Conv1d
__all__ += ['convert_Conv1d']
# Conv2d
__all__ += ['convert_Conv2d']
# conv2d
__all__ += ['convert_conv2d']
# ConvTranspose1d
__all__ += ['convert_ConvTranspose1d']
# ConvTranspose2d
__all__ += ['convert_ConvTranspose2d']
# div
__all__ += ['convert_div', 'convert_rdiv']
# exview
__all__ += ['convert_exview']
# flatten
__all__ += ['convert_flatten']
# floor_divide
__all__ += ['convert_floor_div', 'convert_rfloor_div']
# full
__all__ += ['convert_full']
# full_like
__all__ += ['convert_full_like']
# gelu
__all__ += ['convert_gelu']
# getitem
__all__ += ['convert_tensor_getitem']
# GRU
__all__ += ['convert_GRU']
# identity
__all__ += ['convert_identity']
# Identity
__all__ += ['convert_Identity']
# index_select
__all__ += ['convert_index_select']
# instance_norm
__all__ += ['convert_instance_norm']
# interpolate_custom
__all__ += ['convert_interpolate']
# LayerNorm
__all__ += ['convert_LayerNorm']
# Linear
__all__ += ['convert_Linear']
# linear
__all__ += ['convert_linear']
# linspace
__all__ += ['convert_linspace']
# logical
__all__ += [
    'convert_and', 'convert_equal', 'convert_greater', 'convert_greaterequal',
    'convert_less', 'convert_lessequal', 'convert_ne', 'convert_or',
    'convert_xor'
]
# LogSoftmax
__all__ += ['convert_LogSoftmax']
# masked_fill
__all__ += ['convert_masked_fill']
# matmul
__all__ += ['convert_matmul']
# max
__all__ += ['convert_max']
# max_pool1d
__all__ += ['convert_max_pool1d']
# max_pool2d
__all__ += ['convert_max_pool2d']
# mean
__all__ += ['convert_mean']
# min
__all__ += ['convert_min']
# mod
__all__ += ['convert_mod']
# mul
__all__ += ['convert_mul']
# narrow
__all__ += ['convert_narrow']
# new_ones
__all__ += ['convert_new_ones']
# new_zeros
__all__ += ['convert_new_zeros']
# normalize
__all__ += ['convert_normalize']
# numel
__all__ += ['convert_numel']
# ones
__all__ += ['convert_ones']
# ones_like
__all__ += ['convert_ones_like']
# repeat
__all__ += ['convert_repeat', 'convert_expand', 'convert_expand_as']
# interpolate_custom
__all__ += ['convert_interpolate']
# unsqueeze
__all__ += ['convert_unsqueeze']
# flip
__all__ += ['convert_flip']
# pad
__all__ += ['convert_pad']
# permute
__all__ += ['convert_permute']
# pixel_shuffle
__all__ += ['convert_pixel_shuffle']
# pow
__all__ += ['convert_pow', 'convert_rpow']
# prelu
__all__ += ['convert_prelu']
# prod
__all__ += ['convert_prod']
# relu
__all__ += ['convert_relu']
# ReLU
__all__ += ['convert_ReLU']
# relu6
__all__ += ['convert_relu6']
# ReLU6
__all__ += ['convert_ReLU6']
# sigmoid
__all__ += ['convert_sigmoid']
# size
__all__ += [
    'convert_intwarper_add', 'convert_intwarper_floordiv',
    'convert_intwarper_mul', 'convert_intwarper_radd',
    'convert_intwarper_rmul', 'convert_intwarper_rsub',
    'convert_intwarper_sub', 'convert_shapewarper_numel', 'convert_size',
    'convert_intwarper_rfloordiv', 'convert_intwarper_pow',
    'convert_intwarper_rpow'
]
# softmax
__all__ += ['convert_softmax']
# split
__all__ += ['convert_split']
# squeeze
__all__ += ['convert_squeeze']
# stack
__all__ += ['convert_stack']
# std
__all__ += ['convert_std']
# sub
__all__ += ['convert_sub', 'convert_rsub']
# sum
__all__ += ['convert_sum']
# t
__all__ += ['convert_t']
# take
__all__ += ['convert_take']
# tanh
__all__ += ['convert_tanh']
# to
__all__ += ['convert_Tensor_to']
# topk
__all__ += ['convert_topk']
# transpose
__all__ += ['convert_transpose']
# unary
__all__ += [
    'convert_abs', 'convert_acos', 'convert_asin', 'convert_atan',
    'convert_ceil', 'convert_cos', 'convert_cosh', 'convert_floor',
    'convert_invert', 'convert_log', 'convert_log2', 'convert_neg',
    'convert_reciprocal', 'convert_sin', 'convert_sinh', 'convert_sqrt',
    'convert_tan'
]
# view
__all__ += ['convert_view']
# view
__all__ += ['convert_view_as']
# where
__all__ += ['convert_Tensor_where', 'convert_where']
# zeros
__all__ += ['convert_zeros']
# zeros_like
__all__ += ['convert_zeros_like']
# meshgrid
__all__ += ['convert_meshgrid']
# roll
__all__ += ['convert_roll']

try:
    # custom plugin support
    from .adaptive_avg_pool1d import convert_adaptive_avg_pool1d
    from .adaptive_max_pool1d import convert_adaptive_max_pool1d
    from .adaptive_avg_pool2d import convert_adaptive_avg_pool2d
    from .adaptive_max_pool2d import convert_adaptive_max_pool2d
    from .AdaptiveAvgPool2d import convert_AdaptiveAvgPool2d
    from .AdaptiveMaxPool2d import convert_AdaptiveMaxPool2d
    from .bmm import convert_bmm
    from .cummax import convert_cummax
    from .cummin import convert_cummin
    from .cumprod import convert_cumprod
    from .cumsum import convert_cumsum
    from .deform_conv2d import convert_deform_conv2d
    from .Embedding import convert_embedding, convert_embedding_forward
    from .gather import convert_gather
    from .grid_sample import convert_grid_sample
    from .GroupNorm import convert_GroupNorm
    from .nms import convert_nms
    from .roi_align import convert_roi_align, convert_RoiAlign
    from .roi_pool import convert_roi_pool, convert_RoIPool
    from .unfold import convert_unfold

    # adaptive_avg_pool1d
    __all__ += ['convert_adaptive_avg_pool1d']
    # adaptive_max_pool1d
    __all__ += ['convert_adaptive_max_pool1d']
    # adaptive_avg_pool2d
    __all__ += ['convert_adaptive_avg_pool2d']
    # adaptive_max_pool2d
    __all__ += ['convert_adaptive_max_pool2d']
    # AdaptiveAvgPool2d
    __all__ += ['convert_AdaptiveAvgPool2d']
    # AdaptiveMaxPool2d
    __all__ += ['convert_AdaptiveMaxPool2d']
    # bmm
    __all__ += ['convert_bmm']
    # cummax
    __all__ += ['convert_cummax']
    # cummin
    __all__ += ['convert_cummin']
    # cumprod
    __all__ += ['convert_cumprod']
    # cumsum
    __all__ += ['convert_cumsum']
    # deform_conv2d
    __all__ += ['convert_deform_conv2d']
    # Embedding
    __all__ += ['convert_embedding', 'convert_embedding_forward']
    # gather
    __all__ += ['convert_gather']
    # grid_sample
    __all__ += ['convert_grid_sample']
    # GroupNorm
    __all__ += ['convert_GroupNorm']
    # nms
    __all__ += ['convert_nms']
    # roi_align
    __all__ += ['convert_roi_align', 'convert_RoiAlign']
    # roi_pool
    __all__ += ['convert_roi_pool', 'convert_RoIPool']
    # unfold
    __all__ += ['convert_unfold']
except Exception:
    print('plugin not found.')
