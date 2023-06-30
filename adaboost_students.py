import numpy as np

from desition_tree_exmpl import DT
from utils.enums import SetTypeOfTask


class Adaboost():
    def __init__(self, M):
        self.M = M
        self.alpha = []
        self.weak_classifiers = []

    def __init_weights(self, N):
        """ initialisation of input variables weights"""
        return np.array([1 / N] * N)

    def update_weights(self, gt, predict, weights, weight_weak_classifiers):
        """ update weights functions DO NOT use loops"""
        bools = np.array(predict != gt)
        bools = np.where(bools == True, 1, bools)
        bools = np.where(weights == False, 0, bools)
        weights = weights * np.exp(weight_weak_classifiers * bools)
        self.weights = weights / np.sum(weights)

    def claculate_error(self, gt, predict, weights):
        """ weak classifier error calculation DO NOT use loops"""
        bool_var = gt != predict
        return np.sum(weights[bool_var])

    def claculate_classifier_weight(self, error):
        """ weak classifier weight calculation DO NOT use loops"""
        return np.log((1 - error) / error)

    def train(self, target, vectors):
        """ train model"""
        self.weights = self.__init_weights(vectors.shape[0])
        for m in range(self.M):
            des_tr = DT(SetTypeOfTask.classification.name)
            des_tr.train(vectors, target, self.weights)
            predictions = des_tr.get_predictions(vectors, False)
            error = self.claculate_error(target, predictions, self.weights)
            alpha = self.claculate_classifier_weight(error)
            self.update_weights(target, predictions, self.weights, alpha)
            self.alpha.append(alpha)
            self.weak_classifiers.append(des_tr)
            if round(float(error), 2) <= 0.52:
                break

    def get_prediction(self, vectors):
        """ adaboost get prediction """
        summa = 0
        for index in range(len(self.weak_classifiers)):
            pred = self.weak_classifiers[index].get_predictions(vectors, False)
            summa += self.alpha[index] * pred
        return np.where(summa > 0, 1, -1)
