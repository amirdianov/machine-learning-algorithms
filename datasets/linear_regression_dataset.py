import numpy as np

from utils.common_functions import read_dataframe_file



class LinRegDataset:
    def __call__(self, dataframe_path: str) -> dict:
        advertising_dataframe = read_dataframe_file(dataframe_path)
        return {
            "inputs": np.asarray(advertising_dataframe["inputs"]),
            "targets": np.asarray(advertising_dataframe["targets"]),
        }

