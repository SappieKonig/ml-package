import torch
import pandas as pd
import numpy as np


from ml_package.types import MassIndex


def select_on_index(data: MassIndex, indices: list[int]):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[indices]
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        return data[indices]
    raise ValueError("Data must be a pandas DataFrame, numpy array, or torch tensor")
