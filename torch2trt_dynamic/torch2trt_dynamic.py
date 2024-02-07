import inspect
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import tensorrt as trt
import torch

from .calibration import (DEFAULT_CALIBRATION_ALGORITHM, DatasetCalibrator,
                          SequenceDataset)
from .shape_converter import ShapeConverter

# UTILITY FUNCTIONS

TORCH_TRT_DTYPE_MAP = {
    torch.bool: trt.bool,
    torch.int8: trt.int8,
    torch.int32: trt.int32,
    torch.float16: trt.float16,
    torch.float32: trt.float32,
}

TRT_TORCH_DTYPE_MAP = {
    trt.bool: torch.bool,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.float16: torch.float16,
    trt.float32: torch.float32,
}


def torch_dtype_to_trt(dtype):
    if dtype in TORCH_TRT_DTYPE_MAP:
        return TORCH_TRT_DTYPE_MAP[dtype]
    else:
        raise TypeError(f'{dtype} is not supported by TensorRT')


def torch_dtype_from_trt(dtype):
    if dtype in TRT_TORCH_DTYPE_MAP:
        return TRT_TORCH_DTYPE_MAP[dtype]
    else:
        raise TypeError(f'{dtype} is not supported by PyTorch')


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError(f'{device} is not supported by TensorRT')


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by PyTorch')


def trt_num_inputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            count += 1
    return count


def trt_num_outputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            count += 1
    return count


def torch_dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask"""
    if not isinstance(dim, tuple):
        dim = (dim, )

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << (d)

    return axes


def add_trt_constant(network, tensor):
    shape = tuple(tensor.shape[1:])
    array = tensor[0].detach().cpu().numpy()
    layer = network.add_constant(shape, array)
    return layer.get_output(0)


def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                if t.dtype == torch.long:
                    dtype = torch.int32
                else:
                    dtype = t.dtype
            else:
                if t.dtype == torch.long:
                    assert (dtype == torch.int32
                            )  # , 'Tensor data types must match')
                else:
                    assert (dtype == t.dtype
                            )  # , 'Tensor data types must match')

    for t in tensors:
        if isinstance(t, float):
            if dtype is None:
                dtype = torch.float
            # else:
            #     assert(dtype == torch.float)
        elif isinstance(t, int):
            if dtype is None:
                dtype = torch.int32
            # else:
            #     assert(dtype == torch.int32)

    # , 'Data type could not be inferred from any item in list')
    assert (dtype is not None)
    return dtype


def trt_(network, *tensors):
    """
    Creates missing TensorRT tensors and adds shuffle layers to make tensors
    broadcastable
    """
    trt_tensors = [None] * len(tensors)

    dtype = check_torch_dtype(*tensors)

    # get broadcast dimension
    broadcast_num_dim = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if not hasattr(t, '_trt'):
                num_dim = len(t.shape)  # don't exclude batch for constants
            else:
                # non-leaf tensors must already have _trt, get shape from that
                num_dim = len(t._trt.shape)
            if num_dim > broadcast_num_dim:
                broadcast_num_dim = num_dim

    for i, t in enumerate(tensors):
        trt_tensor = None

        # GET TRT TENSOR (OR CREATE TRT CONSTANT)

        is_const = False
        # get tensor w/ _trt
        if isinstance(t, torch.Tensor) and hasattr(t, '_trt'):
            trt_tensor = t._trt

        # or... add constant for leaf tensor w/o _trt
        elif isinstance(t, torch.Tensor) and not hasattr(t, '_trt'):
            # add leaf tensor
            # don't exclude batch when adding constants...?
            is_const = True
            shape = tuple(t.shape)
            weight = t.detach().cpu().numpy()
            if weight.dtype == np.float64:
                weight = weight.astype(np.float32)
            elif weight.dtype == np.int64:
                weight = weight.astype(np.int32)
            t._trt = network.add_constant(shape, weight).get_output(0)
            trt_tensor = t._trt
        elif isinstance(t, int) and hasattr(t, '_trt'):
            # Int warper
            trt_tensor = t._trt
            trt_dtype = torch_dtype_to_trt(dtype)
            trt_tensor = trt_cast(network, trt_tensor, trt_dtype)

        # or... add constant for scalar primitive
        elif isinstance(t, float) or isinstance(t, int):
            is_const = True
            shape = (1, )  # * broadcast_num_dim
            scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
            trt_tensor = network.add_constant(shape, scalar).get_output(0)

        assert (trt_tensor is not None)

        # MAKE TRT TENSOR BROADCASTABLE IF IT IS NOT ALREADY

        if len(trt_tensor.shape) < broadcast_num_dim:
            if is_const:
                # append 1 size dims to front
                diff = broadcast_num_dim - len(trt_tensor.shape)
                shape = tuple([1] * diff + list(trt_tensor.shape))
                layer = network.add_shuffle(trt_tensor)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)
            else:
                diff = broadcast_num_dim - len(trt_tensor.shape)
                shape = (diff, )
                scalar = torch.ones(shape, dtype=torch.int32).cpu().numpy()
                trt_ones = network.add_constant(shape, scalar).get_output(0)
                trt_shape = tensor_trt_get_shape_trt(network, trt_tensor)
                trt_shape = network.add_concatenation([trt_ones, trt_shape
                                                       ]).get_output(0)
                layer = network.add_shuffle(trt_tensor)
                layer.set_input(1, trt_shape)
                trt_tensor = layer.get_output(0)

        trt_tensors[i] = trt_tensor

    if len(trt_tensors) == 1:
        return trt_tensors[0]
    else:
        return tuple(trt_tensors)


def slice_shape_trt(network, shape_trt, start=0, size=None, stride=1):
    shape_trt_dim = shape_trt.shape[0]
    if start == 0 and stride == 1 and (size is None or size == shape_trt_dim):
        return shape_trt

    if start >= shape_trt_dim:
        return None

    if size == 0:
        return None

    if size is None:
        size = shape_trt_dim - start

    return network.add_slice(shape_trt, [start], [size],
                             [stride]).get_output(0)


def tensor_trt_get_shape_trt(network,
                             tensor_trt,
                             start=0,
                             size=None,
                             stride=1):
    shape_trt = network.add_shape(tensor_trt).get_output(0)
    return slice_shape_trt(network, shape_trt, start, size, stride)


def trt_cast(network, val_trt, data_type):
    if isinstance(data_type, trt.DataType):
        # zeros_type = torch_dtype_from_trt(data_type)
        pass
    else:
        # zeros_type = data_type
        data_type = torch_dtype_to_trt(data_type)
    origin_dtype = val_trt.dtype

    if origin_dtype == data_type:
        return val_trt

    layer = network.add_identity(val_trt)
    layer.set_output_type(0, data_type)
    val_trt = layer.get_output(0)
    val_trt.shape  # trick to enable type cast, I have no idea why...

    return val_trt


def convert_with_args(ctx, convert_func, args, kw_args, returns):
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    old_return = ctx.method_return

    ctx.method_args = args
    ctx.method_kwargs = kw_args
    ctx.method_return = returns
    convert_func(ctx)

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
    ctx.method_return = old_return


# CONVERSION REGISTRY AND HOOKS

CONVERTERS = {}


def get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default


def attach_converter(ctx, method, converter, method_str):
    """Gets a function that executes PyTorch method and TensorRT converter"""
    global DUMMY_CONVERTERS

    def wrapper(*args, **kwargs):
        skip = True

        # check if another (parent) converter has lock
        if not ctx.lock:
            if converter['is_real']:
                ctx.lock = True  # only real converters can acquire lock
            skip = False

        # run original method
        outputs = method(*args, **kwargs)

        if not skip:
            ctx.method_args = args
            ctx.method_kwargs = kwargs
            ctx.method_return = outputs
            ctx.method_str = method_str

            #             print('%s' % (converter.__name__,))
            converter['converter'](ctx)
            outputs = ctx.method_return

            # convert to None so conversion will fail for unsupported layers
            ctx.method_args = None
            ctx.method_kwargs = None
            ctx.method_return = None
            ctx.lock = False

        return outputs

    return wrapper


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, method, converter):
        self.ctx = ctx
        self.method_str = method
        self.converter = converter

    def _set_method(self, method):
        exec('%s = method' % self.method_str)

    def __enter__(self):
        if not self.method_str.startswith('torch.'):
            module_name = self.method_str.split('.')[0]
            try:
                exec('import ' + module_name, globals())
            except Exception:
                print('module {} not found.'.format(module_name))
        try:
            self.method_impl = eval(self.method_str)
        except AttributeError:
            self.method_impl = None

        if self.method_impl:
            self._set_method(
                attach_converter(self.ctx, self.method_impl, self.converter,
                                 self.method_str))

    def __exit__(self, type, val, tb):
        if self.method_impl:
            self._set_method(self.method_impl)


class ConversionContext(object):

    def __init__(self, network, converters=CONVERTERS):
        self.network = network
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.hooks = [
            ConversionHook(self, method, converter)
            for method, converter in converters.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def add_inputs(self, torch_inputs: Dict, shape_ranges: Dict):

        def __get_input_shape(shape_range: Dict):
            min_shape = np.array(shape_range['min'])
            opt_shape = np.array(shape_range['opt'])
            max_shape = np.array(shape_range['max'])
            eq_mask = (min_shape == opt_shape) & (opt_shape == max_shape)
            input_shape = np.where(eq_mask, opt_shape, -1)
            return tuple(input_shape.tolist())

        self.input_names = list(torch_inputs.keys())

        for name, tensor in torch_inputs.items():
            if hasattr(tensor, '_trt'):
                continue
            input_shape = __get_input_shape(shape_ranges[name])

            trt_tensor = self.network.add_input(
                name=name,
                shape=input_shape,
                dtype=torch_dtype_to_trt(tensor.dtype),
            )
            trt_tensor.location = torch_device_to_trt(tensor.device)
            tensor._trt = trt_tensor

    def mark_outputs(self, torch_outputs):
        if isinstance(torch_outputs, torch.Tensor):
            output_names = ['_output']
            torch_outputs = {output_names[0]: torch_outputs}
            output_type = 'tensor'
        elif isinstance(torch_outputs, Sequence):
            output_names = [
                '_output_%d' % i for i in range(len(torch_outputs))
            ]
            torch_outputs = {
                name: output
                for name, output in zip(output_names, torch_outputs)
            }
            output_type = 'list'
        elif isinstance(torch_outputs, Dict):
            output_names = list(torch_outputs.keys())
            output_type = 'dict'
        else:
            raise TypeError('Unsupported output type: '
                            f'{type(torch_outputs)}')

        self.output_names = output_names
        self.output_type = output_type

        for name, tensor in torch_outputs.items():
            trt_tensor = tensor._trt
            trt_tensor.name = name
            trt_tensor.location = torch_device_to_trt(tensor.device)
            self.network.mark_output(trt_tensor)

        return self.output_names, self.output_type


@dataclass
class TRTModuleMeta:
    signature: inspect.Signature
    input_names: Sequence[str]
    output_names: Sequence[str]
    output_type: str


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, meta: TRTModuleMeta = None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        if engine is not None:
            self._build_module(engine, meta)

    def _build_module(self, engine, meta: TRTModuleMeta):
        assert engine is not None
        assert meta is not None
        self.engine = engine
        self.meta = meta
        self.update_context()

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'meta'] = self.meta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']
        self.meta = state_dict[prefix + 'meta']

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.update_context()

    def update_context(self):
        self.context = self.engine.create_execution_context()

    @property
    def input_names(self):
        return self.meta.input_names

    @property
    def output_names(self):
        return self.meta.output_names

    @property
    def signature(self):
        return self.meta.signature

    @property
    def output_type(self):
        return self.meta.output_type

    def _check_input_shape(self, inputs: Dict):

        def __check_range(name, shape, min_shape, max_shape):
            shape = np.array(shape)
            min_shape = np.array(min_shape)
            max_shape = np.array(max_shape)
            if not (min_shape <= shape).all():
                raise ValueError(f'input <{name}> shape: {shape} '
                                 f'is less than min shape: {min_shape}')
            if not (shape <= max_shape).all():
                raise ValueError(f'input <{name}> shape: {shape} '
                                 f'is greater than max shape: {max_shape}')

        for name, tensor in inputs.items():
            shape = tensor.shape
            input_shapes = self.engine.get_profile_shape(0, name)
            min_shape, opt_shape, max_shape = input_shapes
            assert len(shape) == len(opt_shape), (
                f'input <{name}> dimension mismatch: ',
                f'expected {len(opt_shape)}, got {len(shape)}')
            __check_range(name, shape, min_shape, max_shape)

    def _bind_inputs(self, *args, **kwargs):
        inputs = self.signature.bind(*args, **kwargs).arguments
        inputs = dict(
            (name, tensor.contiguous()) for name, tensor in inputs.items())
        self._check_input_shape(inputs)
        return inputs

    def forward(self, *args, **kwargs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        def __setup_inputs(inputs: Dict, bindings: Sequence):
            for input_name, tensor in inputs.items():
                idx = self.engine.get_binding_index(input_name)
                self.context.set_binding_shape(idx, tuple(tensor.shape))
                bindings[idx] = tensor.data_ptr()

        def __setup_outputs(bindings: Sequence):
            outputs = dict()
            for i, output_name in enumerate(self.output_names):
                idx = self.engine.get_binding_index(output_name)
                dtype = torch_dtype_from_trt(
                    self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))
                device = torch_device_from_trt(self.engine.get_location(idx))
                output = torch.empty(size=shape, dtype=dtype, device=device)
                outputs[output_name] = output
                bindings[idx] = output.data_ptr()
            return outputs

        def __get_return_value(outputs: Dict):
            if self.output_type == 'tensor':
                return outputs['_output']
            elif self.output_type == 'list':
                return [outputs[name] for name in self.output_names]
            elif self.output_type == 'dict':
                return outputs
            else:
                raise TypeError('Unsupported output type: '
                                f'{self.output_type}')

        inputs = self._bind_inputs(*args, **kwargs)
        __setup_inputs(inputs, bindings)
        outputs = __setup_outputs(bindings)

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return __get_return_value(outputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


@dataclass
class BuildEngineConfig:
    shape_ranges: Dict = None
    pool: trt.MemoryPoolType = trt.MemoryPoolType.WORKSPACE
    pool_size: int = 0
    fp16: bool = False
    int8: bool = False
    int8_calib_dataset: Any = None
    int8_calib_algorithm: trt.CalibrationAlgoType = None
    int8_batch_size: int = 1
    int8_cache_file: str = None
    int8_calibrator: Any = None

    def __post_init__(self):
        if self.int8_calib_algorithm is None:
            self.int8_calib_algorithm = DEFAULT_CALIBRATION_ALGORITHM


def _default_shape_ranges(inputs: Dict):
    shape_ranges = dict()
    for name, tensor in inputs.items():
        shape_ranges[name] = dict(
            min=tuple(tensor.shape),
            opt=tuple(tensor.shape),
            max=tuple(tensor.shape))
    return shape_ranges


def build_network(builder: trt.Builder,
                  func: Any,
                  inputs: Dict,
                  config: BuildEngineConfig = None):
    """build trt network"""
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    if config is None:
        config = BuildEngineConfig()

    shape_ranges = config.shape_ranges
    if shape_ranges is None:
        shape_ranges = _default_shape_ranges(inputs)

    with ShapeConverter(), ConversionContext(network) as ctx:
        ctx.add_inputs(inputs, shape_ranges)
        outputs = func(**inputs)
        output_names, output_type = ctx.mark_outputs(outputs)
        torch.cuda.empty_cache()

    signature = inspect.signature(func)
    input_names = list(inputs.keys())
    module_meta = TRTModuleMeta(
        signature=signature,
        input_names=input_names,
        output_names=output_names,
        output_type=output_type)
    return network, module_meta


def build_engine(func: Any,
                 inputs: Dict,
                 config: BuildEngineConfig = None,
                 log_level: trt.Logger = trt.Logger.ERROR):
    """build TensorRT Engine"""

    def __make_profile(builder: trt.Builder, shape_ranges: Dict):
        profile = builder.create_optimization_profile()
        for name, shape_range in shape_ranges.items():
            min_shape = shape_range['min']
            opt_shape = shape_range['opt']
            max_shape = shape_range['max']
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        return profile

    def __setup_fp16(builder_config):
        if config.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

    def __setup_int8(builder_config, profile):
        if not config.int8:
            return
        builder_config.set_flag(trt.BuilderFlag.INT8)
        if config.int8_calibrator is not None:
            builder_config.int8_calibrator = config.int8_calibrator
        else:
            int8_calib_dataset = config.int8_calib_dataset
            if int8_calib_dataset is None:
                int8_calib_dataset = SequenceDataset([inputs] * 10)
            builder_config.int8_calibrator = DatasetCalibrator(
                int8_calib_dataset,
                batch_size=config.int8_batch_size,
                cache_file=config.int8_cache_file,
                algorithm=config.int8_calib_algorithm)
        builder_config.set_calibration_profile(profile)

    def __make_builder_config(builder: trt.Builder, shape_ranges: Dict):
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(config.pool, config.pool_size)
        profile = __make_profile(builder, shape_ranges)
        builder_config.add_optimization_profile(profile)

        __setup_fp16(builder_config)
        __setup_int8(builder_config, profile)
        return builder_config

    if config is None:
        config = BuildEngineConfig()

    shape_ranges = config.shape_ranges
    if shape_ranges is None:
        shape_ranges = _default_shape_ranges(inputs)

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    network, module_meta = build_network(builder, func, inputs, config=config)
    builder_config = __make_builder_config(builder, shape_ranges)
    engine = builder.build_engine(network, builder_config)

    if engine is None:
        raise RuntimeError('Failed to build TensorRT engine')
    return engine, module_meta


def func2trt(func: Any,
             args: Sequence = None,
             kwargs: Dict = None,
             config: BuildEngineConfig = None,
             log_level: trt.Logger = trt.Logger.ERROR):
    """convert callable object to TensorRT module"""

    def __bind_inputs(signature: inspect.Signature):
        nonlocal args, kwargs
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        return signature.bind(*args, **kwargs).arguments

    signature = inspect.signature(func)
    inputs = __bind_inputs(signature)
    engine, module_meta = build_engine(func, inputs, config, log_level)

    trt_module = TRTModule(engine, module_meta)
    return trt_module


def module2trt(module: Any,
               args: Sequence = None,
               kwargs: Dict = None,
               config: BuildEngineConfig = None,
               log_level: trt.Logger = trt.Logger.ERROR):
    """convert torch.nn.Module to TensorRT module"""
    return func2trt(
        module.forward, args, kwargs, config=config, log_level=log_level)


def torch2trt_dynamic(module,
                      inputs,
                      input_names=None,
                      output_names=None,
                      log_level=trt.Logger.ERROR,
                      max_batch_size=1,
                      fp16_mode=False,
                      max_workspace_size=0,
                      opt_shape_param=None,
                      strict_type_constraints=False,
                      keep_network=True,
                      int8_mode=False,
                      int8_calib_dataset=None,
                      int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM):
    print('Warning, torch2trt_dynamic is deprecated, use module2trt instead')

    signature = inspect.signature(module.forward)
    input_names = signature.parameters.keys()
    input_names = list(input_names)[:len(inputs)]

    shape_ranges = None
    if opt_shape_param is not None:
        shape_ranges = dict()
        for i, param in enumerate(opt_shape_param):
            name = input_names[i]
            min_shape, opt_shape, max_shape = param
            shape_ranges[name] = dict(
                min=min_shape, opt=opt_shape, max=max_shape)

    config = BuildEngineConfig(
        shape_ranges=shape_ranges,
        pool_size=max_workspace_size,
        fp16=fp16_mode,
        int8=int8_mode,
        int8_calib_dataset=int8_calib_dataset,
        int8_calib_algorithm=int8_calib_algorithm,
        int8_batch_size=max_batch_size,
    )
    return module2trt(module, args=inputs, config=config, log_level=log_level)


# DEFINE ALL CONVERSION FUNCTIONS


def tensorrt_converter(method, is_real=True):

    def register_converter(converter):
        CONVERTERS[method] = {'converter': converter, 'is_real': is_real}
        return converter

    return register_converter
