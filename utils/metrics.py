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


def count_metrics(metrics):
    TP, TN, FP, FN = metrics[(1, 1)], metrics[(-1, -1)], metrics[(-1, 1)], metrics[(1, -1)]
    metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    metrics['precision'] = TP / (TP + FP)
    metrics['recall'] = TP / (TP + FN)
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
            metrics['precision'] + metrics['recall']) if metrics['precision'] + metrics['recall'] != 0 else 0
    metrics['a'] = FP / (TN + FP)
    metrics['b'] = FN / (TP + FN)
    return metrics


def count_section_metrics(gt, predict):
    metrics = {
        # TrueNeg
        (-1, -1): 0,
        # FalsePos
        (-1, 1): 0,
        # FalseNeg
        (1, -1): 0,
        # TruePos
        (1, 1): 0
    }
    df = np.column_stack((gt, predict))
    for pair in df:
        metrics[(pair[0], pair[1])] += 1
    return count_metrics(metrics)
