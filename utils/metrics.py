import numpy as np
from sklearn.metrics import confusion_matrix


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    pass


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    # TODO calculate accuracy
    pass


def conf_matrix(targets, model_confidence):
    # TODO build confusion matrix
    return confusion_matrix(list(targets), list(model_confidence))
