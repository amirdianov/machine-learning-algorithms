import numpy as np

from config.conf import cfg
from datasets.base_dataset_classes import BaseDataset
from datasets.digits_dataset import Digits
from datasets.wine_dataset import Wine
from desition_tree_exmpl import DT
from utils.common_functions import read_dataframe_file
from utils.enums import SetType, SetTypeOfTask
from utils.metrics import accuracy, conf_matrix, MSE

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)

des_tr = DT(SetTypeOfTask.classification.name)
des_tr.train(train['inputs'], train['targets'])
predictions_valid = des_tr.get_predictions(valid['inputs'])
accuracy_valid = accuracy(predictions_valid, valid['targets'])
conf_matrix_valid = conf_matrix(predictions_valid, valid['targets'])
predictions_test = des_tr.get_predictions(test['inputs'])
accuracy_test = accuracy(predictions_test, test['targets'])
conf_matrix_test = conf_matrix(predictions_test, test['targets'])
print(accuracy_valid, conf_matrix_valid, accuracy_test, conf_matrix_test)


train = Wine(cfg)(SetType.train)
valid = Wine(cfg)(SetType.valid)
test = Wine(cfg)(SetType.test)
des_tr = DT(SetTypeOfTask.regression.name)
des_tr.train(train['inputs'], train['targets'])
predictions_valid = des_tr.get_predictions(valid['inputs'])
print(MSE(predictions_valid, valid['targets']))

