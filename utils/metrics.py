import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Todo calculate loss of your model without loops"""
    return sum((predictions - targets.T) ** 2) / len(targets)
