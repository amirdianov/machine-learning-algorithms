import os

from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = os.path.basename("linear_regression_dataset.csv")
# TODO list of basis functions
cfg.base_functions = [lambda x: x**i for i in range(1000)]
