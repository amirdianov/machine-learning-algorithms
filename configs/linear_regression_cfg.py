import os

from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = os.path.basename("linear_regression_dataset.csv")
# list of basis functions in execute file

max_degree = 100
cfg.base_functions = [lambda x, degree=i: x**degree for i in range(1, 1 + max_degree)]
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
