import os

from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = os.path.basename("linear_regression_dataset.csv")
# TODO list of basis functions
cfg.base_functions = []
