import numpy as np
from sklearn.metrics import confusion_matrix


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """calculate loss of your model without loops"""
    return np.square(np.subtract(targets, predictions)).mean()


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    # calculate accuracy
    count_true = 0
    massive = zip(predictions, targets)
    count_true = sum([1 for pair in massive if pair[0] == pair[1]])
    return count_true / len(predictions)


def conf_matrix(predict, targets):
    # build confusion matrix
    return confusion_matrix(list(targets), list(predict))
