import random

from config.conf import cfg
from datasets.digits_dataset import Digits
from random_forest_exmpl import RandomForest
from utils.enums import SetType
from utils.metrics import accuracy, conf_matrix
from utils.visualisation import Visualisation

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)
models = []
best_model = [[0]]
for _ in range(3):
    M = 3
    # M = random.randint(cfg.M_left, cfg.M_right)
    L1 = random.randint(cfg.L1_left, cfg.L1_right)
    L2 = random.randint(cfg.L2_left, cfg.L2_right)
    rand_fr_tr = RandomForest(nb_trees=M, max_depth=8, min_entropy=0.05, min_elem=25, max_nb_dim_to_check=L1,
                              max_nb_thresholds=L2)
    rand_fr_tr.train(train['inputs'], train['targets'], 10)
    predictions_valid = rand_fr_tr.get_prediction(valid['inputs'])
    accuracy_valid = accuracy(predictions_valid, valid['targets'])
    predictions_test = rand_fr_tr.get_prediction(test['inputs'])
    accuracy_test = accuracy(predictions_test, test['targets'])

    models.append((rand_fr_tr, accuracy_valid, accuracy_test, M, L1, L2))
    if accuracy_valid > best_model[0][0]:
        best_model[0] = [accuracy_valid, predictions_valid]
models = sorted(models, key=lambda x: x[1])
models_best = models[-10::]
print(models_best)
conf_matrix_valid = conf_matrix(best_model[0][1], valid['targets'])
print(conf_matrix_valid)
Visualisation.visualise_best_models(models_best)
