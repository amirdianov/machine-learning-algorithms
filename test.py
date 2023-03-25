from config.logistic_regression_config import cfg
from datasets.digits_dataset import Digits

a = Digits(cfg)
print(a.inputs)
