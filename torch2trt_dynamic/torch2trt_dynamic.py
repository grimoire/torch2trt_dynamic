import numpy as np
import tensorrt as trt
import torch

from .calibration import (DEFAULT_CALIBRATION_ALGORITHM, DatasetCalibrator,
                          TensorBatchDataset)
from .shape_converter import ShapeConverter

# UTILITY FUNCTIONS


def torch_dtype_to_trt(dtype):
    if dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


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
            layer = network.add_identity(trt_tensor)
            layer.set_output_type(0, trt_dtype)
            trt_tensor = layer.get_output(0)

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
        zeros_type = torch_dtype_from_trt(data_type)
    else:
        zeros_type = data_type
        data_type = torch_dtype_to_trt(data_type)
    origin_dtype = val_trt.dtype

    if origin_dtype == data_type:
        return val_trt

    layer = network.add_identity(val_trt)
    layer.set_output_type(0, data_type)
    val_trt = layer.get_output(0)

    if len(val_trt.shape) == 1:
        layer = network.add_elementwise(
            trt_(network, torch.zeros((1, ), dtype=zeros_type)), val_trt,
            trt.ElementWiseOperation.SUM)
        layer.set_output_type(0, data_type)
        val_trt = layer.get_output(0)

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
                print("module {} not found.".format(module_name))
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

    def add_inputs(self, torch_inputs, names=None, opt_shape_param=None):
        if names is None:
            names = ['input_%d' % i for i in range(len(torch_inputs))]
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if not hasattr(torch_input, '_trt'):
                if opt_shape_param is not None:
                    # input_shape = (-1,)*len(torch_input.shape)
                    input_shape = []
                    for idx in range(len(torch_input.shape)):
                        if opt_shape_param[i][0][idx] == opt_shape_param[i][1][
                                idx] == opt_shape_param[i][2][idx]:
                            input_shape.append(torch_input.shape[idx])
                        else:
                            input_shape.append(-1)
                    input_shape = tuple(input_shape)
                else:
                    input_shape = tuple(torch_input.shape)

                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=input_shape,
                    dtype=torch_dtype_to_trt(torch_input.dtype),
                )
                trt_tensor.location = torch_device_to_trt(torch_input.device)
                torch_input._trt = trt_tensor

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = ['output_%d' % i for i in range(len(torch_outputs))]
        self.output_names = names

        for i, torch_output in enumerate(torch_outputs):
            trt_tensor = torch_output._trt
            trt_tensor.name = names[i]
            trt_tensor.location = torch_device_to_trt(torch_output.device)
            # if not support_dynamic_shape:
            #     trt_tensor.dtype = torch_dtype_to_trt(torch_output.dtype)
            self.network.mark_output(trt_tensor)


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)

            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


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

    inputs_in = inputs

    # copy inputs to avoid modifications to source data
    inputs = [tensor.clone() for tensor in inputs]

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    with ShapeConverter(), ConversionContext(network) as ctx:

        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs, )
        ctx.add_inputs(inputs, input_names, opt_shape_param)

        outputs = module(*inputs)

        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs, )
        ctx.mark_outputs(outputs, output_names)

        torch.cuda.empty_cache()

        builder.max_workspace_size = max_workspace_size
        builder.max_batch_size = max_batch_size
        builder.strict_type_constraints = strict_type_constraints

        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        profile = builder.create_optimization_profile()

        if input_names is None:
            input_names = ['input_%d' % i for i in range(len(inputs))]
        for input_index, input_tensor in enumerate(inputs):
            if opt_shape_param is not None:
                min_shape = tuple(opt_shape_param[input_index][0][:])
                opt_shape = tuple(opt_shape_param[input_index][1][:])
                max_shape = tuple(opt_shape_param[input_index][2][:])
            else:
                opt_shape = tuple(input_tensor.shape)
                min_shape = opt_shape
                max_shape = opt_shape
            profile.set_shape(input_names[input_index], min_shape, opt_shape,
                              max_shape)
        config.add_optimization_profile(profile)

    if fp16_mode:
        builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)

        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = DatasetCalibrator(
            input_names,
            profile,
            inputs_in,
            int8_calib_dataset,
            batch_size=opt_shape[0],
            algorithm=int8_calib_algorithm)
        config.set_calibration_profile(profile)
        builder.int8_mode = int8_mode
        builder.int8_calibrator = config.int8_calibrator

    engine = builder.build_engine(network, config)

    module_trt = TRTModule(engine, ctx.input_names, ctx.output_names)

    if keep_network:
        module_trt.network = network

    return module_trt


# DEFINE ALL CONVERSION FUNCTIONS


def tensorrt_converter(method, is_real=True):

    def register_converter(converter):
        CONVERTERS[method] = {'converter': converter, 'is_real': is_real}
        return converter

    return register_converter
