import os

from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = os.path.basename("linear_regression_dataset.csv")
# list of basis functions in execute file
cfg.base_functions = []
