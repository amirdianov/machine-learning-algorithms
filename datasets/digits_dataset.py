import numpy as np
from easydict import EasyDict
from sklearn.datasets import load_digits
from utils.enums import SetType

from datasets.base_dataset_classes import BaseDataset


class Digits(BaseDataset):

    def __init__(self, cfg: EasyDict):

        super(Digits, self).__init__(cfg.train_set_percent, cfg.valid_set_percent)
        digits = load_digits()

        # define properties
        self.inputs = digits.data
        self.targets = digits.target

        # divide into sets
        self.divide_into_sets()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    def __call__(self, set_type: SetType) -> dict:
        inputs, targets = getattr(self, f'inputs_{set_type.name}'), getattr(self, f'targets_{set_type.name}')
        return {'inputs': inputs,
                'targets': targets}
