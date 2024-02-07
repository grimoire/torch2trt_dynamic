# torch2trt dynamic

This is a branch of [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) with dynamic input support.

## Usage

Here are some examples

### Convert

```python
from torch2trt_dynamic import module2trt, BuildEngineConfig
import torch
from torchvision.models import resnet18

# create some regular pytorch model...
model = resnet18().cuda().eval()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
    config = BuildEngineConfig(
        shape_ranges=dict(
            x=dict(
                min=(1, 3, 224, 224),
                opt=(2, 3, 224, 224),
                max=(4, 3, 224, 224),
            )
        ))
    trt_model = module2trt(
        model,
        args=[x],
        config=config)
```

### Execute

We can execute the returned `TRTModule` just like the original PyTorch model

```python
x = torch.rand(1, 3, 224, 224).cuda()
with torch.no_grad():
    y = model(x)
    y_trt = trt_model(x)

# check the output against PyTorch
torch.testing.assert_close(y, y_trt)
```

### Save and load

We can save the model as a ``state_dict``.

```python
torch.save(trt_model.state_dict(), 'my_engine.pth')
```

We can load the saved model into a ``TRTModule``

```python
from torch2trt_dynamic import TRTModule

trt_model = TRTModule()
trt_model.load_state_dict(torch.load('my_engine.pth'))
```

## Setup

To install without compiling plugins, call the following

```bash
git clone https://github.com/grimoire/torch2trt_dynamic.git torch2trt_dynamic
cd torch2trt_dynamic
pip install .
```

### Set plugins(optional)

Some layers such as `GN` need c++ plugins. Install the plugin project below

[amirstan_plugin](https://github.com/grimoire/amirstan_plugin)

**DO NOT FORGET** to export the environment variable `AMIRSTAN_LIBRARY_PATH`

## How to add (or override) a converter

Here we show how to add a converter for the ``ReLU`` module using the TensorRT Python API.

```python
import tensorrt as trt
from torch2trt_dynamic import tensorrt_converter

@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)
```

The converter takes one argument, a ``ConversionContext``, which will contain
the following

* ``ctx.network`` - The TensorRT network that is being constructed.

* ``ctx.method_args`` - Positional arguments that were passed to the specified PyTorch function.  The ``_trt`` attribute is set for relevant input tensors.
* ``ctx.method_kwargs`` - Keyword arguments that were passed to the specified PyTorch function.
* ``ctx.method_return`` - The value returned by the specified PyTorch function.  The converter must set the ``_trt`` attribute where relevant.

Please see [this folder](torch2trt_dynamic/converters) for more examples.
