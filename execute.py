import numpy as np

from dataset.sportsmans_height import Sportsmanheight
from model.counter_of_metrics import CounterOfMetrics
from model.simple_classifier import Classifier
from model.visualisation import Visualisation

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']
table = np.column_stack([predictions, gt, np.zeros((len(gt)))])
sorted_table = sorted(table, key=lambda x: x[0], reverse=True)
metrics = CounterOfMetrics(sorted_table)()
Visualisation.visualise_pr_curve(metrics)
Visualisation.visualise_roc_curve(metrics)
