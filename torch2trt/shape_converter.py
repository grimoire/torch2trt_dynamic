import torch


def get_tensor_shape(self):
    return self.size()


old_get_attribute = torch.Tensor.__getattribute__
def new_getattribute__(self, name):
    if name is 'shape':
        return get_tensor_shape(self)
    else:
        return old_get_attribute(self, name)

class ShapeConverter:
    def __init__(self):
        pass

    def __enter__(self):
        torch.Tensor.__getattribute__ = new_getattribute__

    def __exit__(self, type, val, tb):
        torch.Tensor.__getattribute__ = old_get_attribute