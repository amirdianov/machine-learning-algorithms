from adaboost_students import Adaboost
from dataset_titanik import Titanic
from utils.metrics import conf_matrix, count_section_metrics

N = 30
model = Adaboost(N)
train_test_data = Titanic('titanik_train_data.csv', 'titanik_test_data.csv')()
model.train(train_test_data['train_target'], train_test_data['train_input'])
predict = model.get_prediction(train_test_data['test_input'])
print(conf_matrix(predict, train_test_data['test_target']))
print(count_section_metrics(train_test_data['test_target'], predict))
