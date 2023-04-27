from adaboost_students import Adaboost
from dataset_titanik import Titanic
from utils.metrics import conf_matrix

N = 30
model = Adaboost(N)
train_test_data = Titanic('titanik_train_data.csv', 'titanik_test_data.csv')()
model.train(train_test_data['train_target'], train_test_data['train_input'])
predict = model.get_prediction(train_test_data['test_input'])
print(conf_matrix(predict, train_test_data['test_target']))
# model
# des_tr = DT(SetTypeOfTask.classification.name)
# des_tr.train(des_tr[], train['targets'])
# predictions_valid = des_tr.get_predictions(valid['inputs'])
# accuracy_valid = accuracy(predictions_valid, valid['targets'])
# conf_matrix_valid = conf_matrix(predictions_valid, valid['targets'])
# predictions_test = des_tr.get_predictions(test['inputs'])
# accuracy_test = accuracy(predictions_test, test['targets'])
# conf_matrix_test = conf_matrix(predictions_test, test['targets'])
# print(accuracy_valid, conf_matrix_valid, accuracy_test, conf_matrix_test)
#
# des_tr = DT(SetTypeOfTask.regression.name)
# des_tr.train(train['inputs'], train['targets'])
# predictions_valid = des_tr.get_predictions(valid['inputs'])
# print(MSE(predictions_valid, valid['targets']))
