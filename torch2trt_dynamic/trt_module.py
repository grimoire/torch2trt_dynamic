import inspect
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import tensorrt as trt
import torch

from .torch_allocator import TorchAllocator

TRT_TORCH_DTYPE_MAP = {
    trt.bool: torch.bool,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.float16: torch.float16,
    trt.float32: torch.float32,
}


def torch_dtype_from_trt(dtype):
    if dtype in TRT_TORCH_DTYPE_MAP:
        return TRT_TORCH_DTYPE_MAP[dtype]
    else:
        raise TypeError(f'{dtype} is not supported by PyTorch')


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by PyTorch')


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

        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.update_context()

    def update_context(self):
        self.context = self.engine.create_execution_context()
        self.allocator = TorchAllocator()
        if hasattr(self.context, 'temporary_allocator'):
            self.context.temporary_allocator = self.allocator

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
            input_shapes = self.engine.get_tensor_profile_shape(name, 0)
            min_shape, opt_shape, max_shape = input_shapes
            assert len(shape) == len(opt_shape), (
                f'input <{name}> dimension mismatch: ',
                f'expected {len(opt_shape)}, got {len(shape)}')
            __check_range(name, shape, min_shape, max_shape)

    def _bind_inputs(self, *args, **kwargs):
        inputs = self.signature.bind(*args, **kwargs).arguments
        inputs = dict(
            (name, tensor.contiguous()) for name, tensor in inputs.items())
        for name, tensor in inputs.items():
            if tensor.dtype == torch.int64:
                tensor = tensor.to(torch.int32)
                inputs[name] = tensor
        self._check_input_shape(inputs)
        return inputs

    def forward(self, *args, **kwargs):

        def __setup_inputs(inputs: Dict):
            for input_name, tensor in inputs.items():
                self.context.set_input_shape(input_name, tuple(tensor.shape))
                self.context.set_tensor_address(input_name, tensor.data_ptr())

        def __setup_outputs():
            outputs = dict()
            for output_name in self.output_names:
                dtype = torch_dtype_from_trt(
                    self.engine.get_tensor_dtype(output_name))
                shape = tuple(self.context.get_tensor_shape(output_name))
                device = torch_device_from_trt(
                    self.engine.get_tensor_location(output_name))
                output = torch.empty(size=shape, dtype=dtype, device=device)
                outputs[output_name] = output
                self.context.set_tensor_address(output_name, output.data_ptr())
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
        device = tuple(inputs.values())[0].device

        with torch.cuda.device(device):
            __setup_inputs(inputs)
            outputs = __setup_outputs()
            self.context.execute_async_v3(
                torch.cuda.current_stream().cuda_stream)
        return __get_return_value(outputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
