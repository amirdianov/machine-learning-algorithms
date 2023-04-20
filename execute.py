from config.conf import cfg
from datasets.digits_dataset import Digits
from random_forest_exmpl import RandomForest
from utils.enums import SetType

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)

rand_fr_tr = RandomForest
rand_fr_tr.train(train['inputs'], train['targets'])

# des_tr = DT(SetTypeOfTask.classification.name)
# des_tr.train(train['inputs'], train['targets'])
# predictions_valid = des_tr.get_predictions(valid['inputs'])
# accuracy_valid = accuracy(predictions_valid, valid['targets'])
# conf_matrix_valid = conf_matrix(predictions_valid, valid['targets'])
# predictions_test = des_tr.get_predictions(test['inputs'])
# accuracy_test = accuracy(predictions_test, test['targets'])
# conf_matrix_test = conf_matrix(predictions_test, test['targets'])
# print(accuracy_valid, conf_matrix_valid, accuracy_test, conf_matrix_test)
#
#
# train = Wine(cfg)(SetType.train)
# valid = Wine(cfg)(SetType.valid)
# test = Wine(cfg)(SetType.test)
# des_tr = DT(SetTypeOfTask.regression.name)
# des_tr.train(train['inputs'], train['targets'])
# predictions_valid = des_tr.get_predictions(valid['inputs'])
# print(MSE(predictions_valid, valid['targets']))
