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

    def divide_into_sets(self):
        # define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework

        df = pd.DataFrame(
            np.concatenate((self.inputs, self.targets), axis=1)
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
        self.inputs_train = np.array(self.inputs_train)
        self.targets_train = np.array(self.targets_train).flatten()
        # self.targets_train = (np.array(self.targets_train)).reshape(self.targets_train.shape[0],)
        self.inputs_valid = np.array(self.inputs_valid)
        self.targets_valid = np.array(self.targets_valid).flatten()
        # self.targets_valid = (np.array(self.targets_valid)).reshape(self.targets_valid.shape[0],)
        self.inputs_test = np.array(self.inputs_test)
        self.targets_test = np.array(self.targets_test).flatten()
        # self.targets_test = (np.array(self.targets_test)).reshape(self.targets_test.shape[0],)

