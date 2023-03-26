import numpy as np

from config.logistic_regression_config import cfg
from datasets.digits_dataset import Digits
from models.logistic_regression_model import LogReg
from utils.enums import SetType
from datasets.base_dataset_classes import BaseClassificationDataset
train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)
log_reg = LogReg(cfg, Digits(cfg).k, Digits(cfg).d)
for row in range(len(train['inputs'])):
    log_reg.train(np.array(train['inputs'])[row], train['targets'].T[row], valid['inputs'], valid['targets'])
