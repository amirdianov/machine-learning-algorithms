import numpy as np

from desition_tree_exmpl import DT
from utils.enums import SetTypeOfTask


class Adaboost():
    def __init__(self, M):
        self.M = M

    def __init_weights(self, N):
        """ initialisation of input variables weights"""
        return np.array([1 / N] * N)

    def update_weights(self, gt, predict, weights, weight_weak_classifiers):
        """ update weights functions DO NOT use loops"""
        pass

    def claculate_error(self, gt, predict, weights):
        """ weak classifier error calculation DO NOT use loops"""
        pass

    def claculate_classifier_weight(self, gt, predict, weights):
        """ weak classifier weight calculation DO NOT use loops"""
        pass

    def train(self, target, vectors):
        """ train model"""
        self.weights = self.__init_weights(vectors.shape[0])
        for m in range(self.M):
            des_tr = DT(SetTypeOfTask.classification.name)

    def get_prediction(self, vectors):
        """ adaboost get prediction """
        pass
