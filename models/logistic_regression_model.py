from typing import Union

import numpy as np
from easydict import EasyDict
from scipy.special import softmax

from utils import metrics


class LogReg:
    BACK_UP = {'target_value_func':[],
               'accuracy_train': [],
               'accuracy_valid': []}

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)
        getattr(self, f'b_init_{cfg.weights_init_type.name}')()

    def weights_init_normal(self, sigma):
        self.weights = np.random.randn(self.k, self.d)

    def b_init_normal(self):
        self.b = np.random.randn(self.k)

    def weights_init_uniform(self, epsilon):
        # TODO init weights with values from uniform distribution BONUS TASK
        pass

    def weights_init_xavier(self, n_in, n_out):
        # TODO Xavier weights initialisation BONUS TASK
        pass

    def weights_init_he(self, n_in):
        # TODO He weights initialisation BONUS TASK
        pass

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        # softmax function realisation
        #  subtract max value of the model_output for numerical stability

        maxim_value = np.max(model_output)
        model_output = model_output - maxim_value
        # sum_variables = sum([e ** model_output[elem] for elem in range(len(model_output))])
        # for i in range(len(model_output)):
        #     model_output[i] = e ** model_output[i] / sum_variables
        return softmax(model_output)

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model confidence (y in lecture)
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model output (z in lecture) using matrix multiplication DONT USE LOOPS
        return self.weights @ inputs.T + self.b

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # calculate gradient for w
        #  slide 10 in presentation
        return (model_confidence - targets).T @ inputs

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # calculate gradient for b
        #  slide 10 in presentation
        return (model_confidence - targets).T @ np.ones(targets.shape[0])

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        # update model weights
        #  slide 8, item 2 in presentation for updating weights
        w_grad = self.__get_gradient_w(inputs, targets,
                                       model_confidence)
        b_grad = self.__get_gradient_b(targets, model_confidence)
        self.weights -= self.cfg.gamma * w_grad
        self.b -= self.cfg.gamma * b_grad

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                onehotencoding_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None,
                                onehotencoding_valid: Union[np.ndarray, None] = None):

        # one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #
        #  update weights
        #   you can add some other steps if you need
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """
        Y_train = []
        Y_valid = []
        for row in range(len(inputs_train)):
            Y_train.append(self.get_model_confidence(inputs_train[row]))
        for row in range(len(inputs_valid)):
            Y_valid.append(self.get_model_confidence(inputs_valid[row]))
        Y_train = np.array(Y_train)
        Y_valid = np.array(Y_valid)
        self.__weights_update(inputs_train, onehotencoding_train, Y_train)
        target_value_result = self.__target_function_value(inputs_train, onehotencoding_train, Y_train)
        train_matrix, train_accuracy = self.__validate(inputs_train, targets_train, Y_train)
        valid_matrix, valid_accuracy = self.__validate(inputs_valid, targets_valid, Y_valid)
        self.BACK_UP['target_value_func'].append([epoch, target_value_result])
        self.BACK_UP['accuracy_train'].append([epoch, train_accuracy])
        self.BACK_UP['accuracy_valid'].append([epoch, valid_accuracy])

        if epoch % 10 == 0:
            print(f'Target func value: {target_value_result}')
            print(f'Epoch:{epoch}. Для train, train_matrix: {train_matrix}, train_accuracy: {train_accuracy}')
            print(f'Epoch:{epoch}. Для valid, valid_matrix: {valid_matrix}, valid_accuracy: {valid_accuracy}')

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               onehotencoding_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None,
                               onehotencoding_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, onehotencoding_train, epoch, inputs_valid,
                                         targets_valid, onehotencoding_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        #  BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray, onehotencoding_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None,
              targets_valid: Union[np.ndarray, None] = None,
              onehotencoding_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                onehotencoding_train,
                                                                                inputs_valid,
                                                                                targets_valid,
                                                                                onehotencoding_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        # target function value calculation
        #  use formula from slide 6 for computational stability
        summa = 0
        for i in range(len(inputs)):
            summa += targets[i] @ np.log(model_confidence[i].T)
        return -summa
        #     k_cls = np.where(targets[i] == 1)
        #     summa += targets[i][k_cls] @ (
        #         np.log(sum([e ** variable for variable in model_confidence[i]]) - model_confidence[i][k_cls]))
        # return summa

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        # metrics calculation: accuracy, confusion matrix
        matrix = metrics.conf_matrix(targets, np.argmax(model_confidence, axis=1))
        accuracy = metrics.accuracy(targets, np.argmax(model_confidence, axis=1))
        return matrix, accuracy

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions
