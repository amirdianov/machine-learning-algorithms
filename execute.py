import numpy as np

from dataset.sportsmans_height import Sportsmanheight
from model.counter_of_metrics import CounterOfMetrics
from model.simple_classifier import Classifier

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']
table = np.column_stack([predictions, gt, np.zeros((len(gt)))])
sorted_table = sorted(table, key=lambda x: x[0], reverse=True)
metrics = CounterOfMetrics(sorted_table)()










# t = 0
# calc_val = []
# while t < len(sorted_table) - 1:
#     sorted_table[t][2] = 1
#     probability = sorted_table[t][0]
#     for number_pair in range(t + 1, len(sorted_table)):
#         if sorted_table[number_pair][0] == probability:
#             sorted_table[number_pair][2] = 1
#             t += 1
#         else:
#             sorted_table[number_pair][2] = 0
#     t += 1
#     df = pd.DataFrame(sorted_table, columns=['conf', 'class', 'predict'])
#     TP = np.column_stack([np.array(df['class'])[:t], np.array(df['predict'][:t])])
#     unique, counts = np.unique(TP, return_counts=True)
#
#     print(dict(zip(unique, counts)))
