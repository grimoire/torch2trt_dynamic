import os
from typing import Tuple

import tensorrt as trt

if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = \
        trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class TensorBatchDataset():

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return [t[idx] for t in self.tensors]


class SequenceDataset():

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


ShapeType = Tuple[int, ...]


class DatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 cache_file: str = None,
                 algorithm=DEFAULT_CALIBRATION_ALGORITHM):
        super(DatasetCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.cache_file = cache_file

        # create buffers that will hold data batches
        self.buffers = dict()
        self.dataset_iter = iter(dataset)

    def get_batch(self, names):
        try:
            inputs = next(self.dataset_iter)
            for name in names:
                tensor = inputs[name]
                if name not in self.buffers:
                    self.buffers[name] = tensor.clone().cuda()
                else:
                    buf = self.buffers[name]
                    assert buf.shape == tensor.shape
                    buf.copy_(tensor)
            return [int(self.buffers[name].data_ptr()) for name in names]
        except StopIteration:
            return list()

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if self.cache_file is None:
            return
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
