import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """calculate loss of your model without loops"""
    return sum((targets.T - predictions) ** 2) / len(targets)
