import random

from config.conf import cfg
from datasets.wine_dataset import Wine
from gradient_boosting import GradientBoosting
from utils.enums import SetType
from utils.metrics import MSE
from utils.visualisation import Visualisation

train = Wine(cfg)(SetType.train)
valid = Wine(cfg)(SetType.valid)
test = Wine(cfg)(SetType.test)
models = []
for _ in range(3):
    M = 5
    a = 1
    model = GradientBoosting(M, a)
    model.train(train['inputs'], train['targets'])
    predictions_valid = model.get_prediction(valid['inputs'])
    MSE_valid = MSE(predictions_valid, valid['targets'])
    predictions_test = model.get_prediction(test['inputs'])
    MSE_test = MSE(predictions_test, test['targets'])
    models.append((model, MSE_valid, MSE_test, M, a))
models = sorted(models, key=lambda x: x[1])
models_best = models[-10::]
print(models_best)

