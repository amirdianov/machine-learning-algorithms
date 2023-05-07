from easydict import EasyDict

from datasets.base_dataset_classes import BaseDataset
from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class Wine(BaseDataset):

    def __init__(self, cfg: EasyDict):
        super(Wine, self).__init__(cfg.train_set_percent, cfg.valid_set_percent)
        df = read_dataframe_file('wine-quality-white-and-red.csv')
        df.loc[df['type'] == 'white', 'type'] = 1
        df.loc[df['type'] == 'red', 'type'] = 0
        # define properties
        self.inputs = df.iloc[:, :-1].to_numpy()
        self.targets = df.iloc[:, -1:].to_numpy()

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
