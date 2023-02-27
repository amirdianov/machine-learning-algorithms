import numpy as np


def func(x):
    return x**2


inputs = np.array([1, 2, 3, 4, 5])
ar = np.ones_like(inputs)
mas = func(inputs)
new = np.append(ar, [mas, mas])
print(new)
