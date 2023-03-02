import numpy as np
import pandas as pd
from easydict import EasyDict
from sklearn.model_selection import train_test_split

from utils.common_functions import read_dataframe_file


class LinRegDataset:
    def __init__(self, cfg: EasyDict):
        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)
        inputs, targets = np.asarray(advertising_dataframe["inputs"]), np.asarray(
            advertising_dataframe["targets"]
        )
        self.__divide_into_sets(
            inputs, targets, cfg.train_set_percent, cfg.valid_set_percent
        )

    def __divide_into_sets(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        train_set_percent: float = 0.8,
        valid_set_percent: float = 0.1,
    ) -> None:
        # define self.inputs_train, self.targets_train, self.inputs_valid,
        # self.targets_valid, self.inputs_test, self.targets_test
        df = pd.DataFrame(
            np.concatenate((inputs.reshape(-1, 1), targets.reshape(-1, 1)), axis=1),
            columns=["inputs", "targets"],
        )
        df_shuffled = df.sample(frac=1, random_state=1)
        inputs_shuffled = df_shuffled["inputs"]
        targets_shuffled = df_shuffled["targets"]
        x_train, x_memory, y_train, y_memory = train_test_split(
            inputs_shuffled, targets_shuffled, train_size=train_set_percent
        )
        x_valid, x_test, y_valid, y_test = train_test_split(
            x_memory,
            y_memory,
            test_size=round(valid_set_percent * 100 / (1 - train_set_percent)) / 100,
        )
        # df = pd.DataFrame(np.concatenate((x_train.to_numpy().reshape(-1, 1), y_train.to_numpy().reshape(-1, 1)), axis=1))
        # print(df)
        # print(x_train.shape,), print(y_train.shape)
        # print(x_valid.shape), print(y_valid.shape)
        # print(x_test.shape), print(y_test.shape)

    def __call__(self) -> dict:
        return {
            "inputs": {
                "train": self.inputs_train,
                "valid": self.inputs_valid,
                "test": self.inputs_test,
            },
            "targets": {
                "train": self.targets_train,
                "valid": self.targets_valid,
                "test": self.targets_test,
            },
        }
