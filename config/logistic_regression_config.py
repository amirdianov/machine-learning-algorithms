import random

from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

# training
cfg.weights_init_type = WeightsInitType.normal
cfg.weights_init_kwargs = {'sigma': 1}

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 100

cfg.lam = 0.01
# TODO threshhold for gradient_descent_difference_norm
cfg.threshold = None