import random

from easydict import EasyDict

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.L1_left, cfg.L1_right = 10, 40
cfg.L2_left, cfg.L2_right = 5, 35
cfg.M_left, cfg.M_right = 5, 5
