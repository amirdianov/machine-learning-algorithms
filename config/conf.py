from easydict import EasyDict

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.L1 = (10, 40)
cfg.L2 = (5, 35)
cfg.M = (5, 20)
