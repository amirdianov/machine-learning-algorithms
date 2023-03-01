import os

from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = os.path.basename("linear_regression_dataset.csv")
# list of basis functions in execute file

cfg.base_functions = []  # TODO list of basis functions
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
