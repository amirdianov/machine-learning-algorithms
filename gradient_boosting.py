import numpy as np

from desition_tree_exmpl import DT
from utils.enums import SetTypeOfTask


class GradientBoosting:
    def __init__(self, M, a):
        self.M = M
        self.a = a
        self.weak_regressions = []

    def init_zero_learner(self, vector, flag=True):
        if flag:
            mean = vector.mean()
            return np.array([mean] * len(vector))
        else:
            mas = []
            for i in vector:
                mas.append(np.mean(i))
            mas = np.array(mas)
            mean = mas.mean()
            return np.array([mean] * len(vector))

    def train(self, input_vector, target_vector):
        y = self.init_zero_learner(target_vector)
        for i in range(1, self.M):
            r = target_vector - y
            des_tr = DT(SetTypeOfTask.regression.name)
            des_tr.train(input_vector, r)
            predictions = des_tr.get_predictions(input_vector, False)
            y = y + self.a * predictions
            self.weak_regressions.append(des_tr)

    def get_prediction(self, input_vector):
        summa = self.init_zero_learner(input_vector, False)
        for index in range(len(self.weak_regressions)):
            pred = self.weak_regressions[index].get_predictions(input_vector, False)
            summa += self.a * pred
        return summa

