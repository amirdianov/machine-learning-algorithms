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
for _ in range(30):
    M = random.randrange(cfg.M_left, cfg.M_right, 1)
    a = round(random.random(), 4)
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
Visualisation.visualise_best_models(models_best)
