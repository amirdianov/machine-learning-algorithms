import numpy as np


class LinearRegression:
    def __init__(self, base_functions: list, reg_coeff: float):
        """init weights using np.random.randn (normal distribution with mean=0 and variance=1)"""
        """we take one more weight, because we need column f0 = 1"""
        self.weights = np.random.randn(len(base_functions) + 1)
        self.base_functions = base_functions
        self.reg_coeff = reg_coeff

    def __pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix with regularization using SVD"""
        tuple_of_matrix = np.linalg.svd(matrix, full_matrices=False)
        sigma_from_tuple = tuple_of_matrix[1]
        matrix_sigma_from_tuple = np.diag(sigma_from_tuple)
        element_not_in_reg = matrix_sigma_from_tuple[0][0]
        condition = (
            np.finfo(float).eps
            * max(matrix_sigma_from_tuple.shape[0], matrix_sigma_from_tuple.shape[1])
            * np.max(matrix_sigma_from_tuple)
        )
        matrix_sigma_from_tuple = np.where(
            matrix_sigma_from_tuple > condition,
            matrix_sigma_from_tuple / (matrix_sigma_from_tuple**2 + self.reg_coeff),
            0,
        )
        matrix_sigma_from_tuple[0][0] = (
            element_not_in_reg / element_not_in_reg**2
            if element_not_in_reg > condition
            else 0
        )
        sigma_plus = matrix_sigma_from_tuple.T
        v = tuple_of_matrix[2].T
        u_t = tuple_of_matrix[0].T
        return v @ sigma_plus @ u_t

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """build Plan matrix using list of lambda functions defined in config. 
        Use only one loop (for base_functions)""" """about f0 - we need column with only ones"""
        matrix = [np.ones_like(inputs)]
        for function in self.base_functions:
            matrix = np.append(matrix, np.array([function(inputs)]), 0)
        return matrix.T

    def __calculate_weights(
        self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray
    ) -> None:
        """calculate weights of the model using formula from the lecture"""
        self.weights = pseudoinverse_plan_matrix @ targets
        return self.weights

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        """calculate prediction of the model (y) using formula from the lecture"""
        return np.dot(plan_matrix, self.weights.T)

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # prepare data
        plan_matrix = self.__plan_matrix(inputs)
        pseudoinverse_plan_matrix = self.__pseudoinverse_matrix(plan_matrix)

        # train process
        self.__calculate_weights(pseudoinverse_plan_matrix, targets)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
