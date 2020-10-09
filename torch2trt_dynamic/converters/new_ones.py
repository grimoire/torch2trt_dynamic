from torch2trt_dynamic.torch2trt_dynamic import *


@tensorrt_converter('torch.Tensor.new_ones')
def convert_new_ones(ctx):
    input = ctx.method_args[0]
    size = get_arg(ctx, 'size', pos=1, default=None)
    dtype = get_arg(ctx, 'dtype', pos=2, default=input.dtype)

    output = ctx.method_return

    if isinstance(size, int):
        size = (size, )

    # check const
    is_const = True
    for s in size:
        if hasattr(s,'_trt'):
            is_const = False
            break

    if is_const:
        # create const value
        output_trt = trt_(ctx.network, output)
    
    else:
        # create fill
        trt_size = []
        for s in size:
            if hasattr(s, '_trt'):
                trt_size.append(s._trt)
            else:
                trt_size.append(trt_(ctx.network, s))
        
        trt_size = ctx.network.add_concatenation(trt_size).get_output(0)

        layer = ctx.network.add_fill(size, trt.FillOperation.RANDOM_UNIFORM)
        layer.set_input(0, trt_size)
        layer.set_input(1, trt_(ctx.network, input.new_tensor(1)))
        layer.set_input(2, trt_(ctx.network, input.new_tensor(1)))

        output_trt = layer.get_output(0)


    data_type = None
    if dtype==torch.float32:
        data_type = trt.DataType.FLOAT
    elif dtype==torch.int32 or dtype==torch.long:
        data_type = trt.DataType.INT32
    elif dtype==torch.bool:
        data_type = trt.DataType.BOOL
    else:
        print("unsupported convert type:{}".format(dtype))
    
    if data_type is not None:
        layer = ctx.network.add_identity(output_trt)
        layer.set_output_type(0, data_type)
        output_trt = layer.get_output(0)

    output._trt = output_trt
