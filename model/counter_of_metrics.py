import pandas as pd


class CounterOfMetrics:
    def __init__(self, sorted_table):
        self.sorted_table = sorted_table
        self.all_metrics = []

    def count_metrics(self, metrics):
        TP, TN, FP, FN = metrics[(1, 1)], metrics[(0, 0)], metrics[(1, 0)], metrics[(0, 1)]
        metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        metrics['precision'] = TP / (TP + FP)
        metrics['recall'] = TP / (TP + FN)
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']) if metrics['precision'] + metrics['recall'] != 0 else 0
        return metrics

    def count_section_metrics(self, t, probability):
        metrics = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0
        }
        t += 1
        while True:
            if self.sorted_table[t][0] == probability:
                self.sorted_table[t][0] = 1
                t += 1
            else:
                break
        df = pd.DataFrame(self.sorted_table, columns=['conf', 'class', 'predict'])
        cl, pr = list(df['class']), list(df['predict'])
        for pair in zip(cl, pr):
            metrics[pair] += 1
        self.all_metrics.append(self.count_metrics(metrics))
        return t

    def __call__(self):
        t = 0
        while t < len(self.sorted_table) - 1:
            self.sorted_table[t][2] = 1
            probability = self.sorted_table[t][0]
            t = self.count_section_metrics(t, probability)
        print(self.all_metrics)
        return self.all_metrics
