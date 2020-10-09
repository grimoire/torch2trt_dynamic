from ..torch2trt_dynamic import *

@tensorrt_converter('torch.linspace')
def convert_linspace(ctx):
    start = get_arg(ctx, 'start', pos=0, default=0)
    end = get_arg(ctx, 'end', pos=1, default=1)
    steps = get_arg(ctx, 'steps', pos=2, default=2)
    dtype = get_arg(ctx, 'dtype', pos=4, default=None)
    
    output = ctx.method_return
    dtype = output.dtype
    if dtype==torch.int64:
        dtype = torch.int32

    # check const
    is_const = True
    is_const = False if hasattr(start, '_trt') or hasattr(end, '_trt') or hasattr(steps, '_trt') else is_const

    if is_const:
        # create const value
        output_trt = trt_(ctx.network, output)
    
    else:
        ## create fill

        # compute shape
        start_trt = trt_(ctx.network, start)
        end_trt = trt_(ctx.network, end)
        steps_trt = trt_(ctx.network, steps)

        length_trt = steps_trt

        # to float
        one_trt = trt_(ctx.network, torch.tensor([1], dtype=torch.float32))
        start_trt = trt_cast(ctx.network, start_trt, trt.DataType.FLOAT)
        end_trt = trt_cast(ctx.network, end_trt, trt.DataType.FLOAT)
        steps_trt = trt_cast(ctx.network, steps_trt, trt.DataType.FLOAT)
        
        # length = (end - start + step - 1) // step
        step_trt = ctx.network.add_elementwise(end_trt, start_trt, trt.ElementWiseOperation.SUB).get_output(0)
        step_div_trt = ctx.network.add_elementwise(steps_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        step_trt = ctx.network.add_elementwise(step_trt, step_div_trt, trt.ElementWiseOperation.DIV).get_output(0)

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
