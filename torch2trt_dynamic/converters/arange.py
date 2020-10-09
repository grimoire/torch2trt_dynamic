from torch2trt_dynamic.torch2trt_dynamic import *

@tensorrt_converter('torch.arange')
def convert_arange(ctx):
    if len(ctx.method_args)==1:
        start = 0
        end = ctx.method_args[0]
        kwargs = ctx.method_kwargs
        step = 1 if 'step' not in kwargs else kwargs['step']
        dtype = None if 'dtype' not in kwargs else kwargs['dtype']
    else:
        start = get_arg(ctx, 'start', pos=0, default=0)
        end = get_arg(ctx, 'end', pos=1, default=1)
        step = get_arg(ctx, 'step', pos=2, default=2)
        dtype = get_arg(ctx, 'dtype', pos=4, default=None)
    
    output = ctx.method_return
    dtype = output.dtype
    if dtype==torch.int64:
        dtype = torch.int32

    # cast float to int if necessory
    if not hasattr(start, '_trt') and start%1==0:
        start = int(start)

    if not hasattr(end, '_trt') and end%1==0:
        end = int(end)

    if not hasattr(step, '_trt') and step%1==0:
        step = int(step)

    # check const
    is_const = True
    is_const = False if hasattr(start, '_trt') or hasattr(end, '_trt') or hasattr(step, '_trt') else is_const
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(step, int):
        is_const = True
        print("warning: dynamic arange with start:{} end:{} step:{}, use constant instead.".format(type(start), type(end), type(step)))

    if is_const:
        # create const value
        output_trt = trt_(ctx.network, output)
    
    else:
        ## create fill

        # compute shape
        start_trt = trt_(ctx.network, start)
        end_trt = trt_(ctx.network, end)
        step_trt = trt_(ctx.network, step)
        one_trt = trt_(ctx.network, torch.tensor([1], dtype=torch.int32))

        # # to float
        # one_trt = trt_(ctx.network, torch.tensor([1], dtype=torch.float32))
        # start_trt = trt_cast(ctx.network, start_trt, trt.DataType.FLOAT)
        # end_trt = trt_cast(ctx.network, end_trt, trt.DataType.FLOAT)
        # step_trt = trt_cast(ctx.network, step_trt, trt.DataType.FLOAT)
        
        # length = (end - start + step - 1) // step
        length_trt = ctx.network.add_elementwise(end_trt, start_trt, trt.ElementWiseOperation.SUB).get_output(0)
        length_trt = ctx.network.add_elementwise(length_trt, step_trt, trt.ElementWiseOperation.SUM).get_output(0)
        length_trt = ctx.network.add_elementwise(length_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        length_trt = ctx.network.add_elementwise(length_trt, step_trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)

        # length to int
        length_trt = trt_cast(ctx.network, length_trt, trt.DataType.INT32)

        # start rank 0
        layer = ctx.network.add_shuffle(start_trt)
        layer.reshape_dims = tuple()
        start_trt = layer.get_output(0)

        layer = ctx.network.add_fill(output.shape, trt.FillOperation.LINSPACE)
        layer.set_input(0, length_trt)
        layer.set_input(1, start_trt)
        layer.set_input(2, step_trt)
        output_trt = layer.get_output(0)

    # cast data type
    data_type = torch_dtype_to_trt(dtype)

    if data_type is not None:
        layer = ctx.network.add_identity(output_trt)
        layer.set_output_type(0, data_type)
        output_trt = layer.get_output(0)

    output._trt = output_trt
