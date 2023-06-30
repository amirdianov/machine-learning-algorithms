from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        # define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework

        df = pd.DataFrame(
            np.concatenate((self.inputs, self.targets.reshape(-1, 1)), axis=1)
        )
        df_shuffled = df.sample(frac=1)
        inputs_shuffled = df_shuffled.iloc[:, :-1]
        targets_shuffled = df_shuffled.iloc[:, -1:]
        self.inputs_train, x_memory, self.targets_train, y_memory = train_test_split(
            inputs_shuffled, targets_shuffled, train_size=self.train_set_percent
        )
        self.inputs_valid, self.inputs_test, self.targets_valid, self.targets_test = train_test_split(
            x_memory,
            y_memory,
            test_size=round(self.valid_set_percent * 100 / (1 - self.train_set_percent)) / 100,
        )

    def normalization(self):
        # write normalization method BONUS TASK
        self.inputs_train = (self.inputs_train - self.inputs_train.min()) / (self.inputs_train.max() - self.inputs_train.min())
        self.inputs_valid = (self.inputs_valid - self.inputs_valid.min()) / (self.inputs_valid.max() - self.inputs_valid.min())
        self.inputs_test = (self.inputs_test - self.inputs_test.min()) / (self.inputs_test.max() - self.inputs_test.min())

    def get_data_stats(self):
        # calculate mean and std of inputs vectors of training set by each dimension
        self.input_std = np.std(np.array(self.inputs_train))
        self.input_mean = np.mean(np.array(self.inputs_train))


    def standartization(self):
        # write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP
        self.inputs_train = (self.inputs_train - self.input_mean) / self.input_std
        self.inputs_valid = (self.inputs_valid - self.input_mean) / self.input_std
        self.inputs_test = (self.inputs_test - self.input_mean) / self.input_std


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # create matrix of onehot encoding vactors for input targets
        #  it is possible to do it without loop
        targets = np.array(targets.iloc[:, 0]).astype(int)
        one_hot_encoding = np.zeros((targets.size, number_classes))
        one_hot_encoding[np.arange(targets.size), targets] = 1
        return one_hot_encoding