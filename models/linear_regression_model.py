import numpy as np

from configs.linear_regression_cfg import cfg as lin_reg_cfg


class LinearRegression:
    def __init__(self, base_functions: list):
        # TODO: init weights using np.random.randn (normal distribution with mean=0 and variance=1)
        self.weights = np.random.normal(0, 1, len(base_functions))
        self.base_functions = base_functions

    @staticmethod
    def __pseudoinverse_matrix(matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix using SVD. Not this homework"""
        pass

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        # TODO build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions)
        matrix = [function(inputs) for function in lin_reg_cfg.base_functions]
        return matrix

    def __calculate_weights(
        self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray
    ) -> None:
        """calculate weights of the model using formula from the lecture. Not this homework"""
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        # TODO calculate prediction of the model (y) using formula from the lecture

        pass

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Not this homework"""
        pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
