from .fill import basic_fill
from .transform import encode_label_transform, one_hot_encode_transform, normalize

import pandas as pd


def tree_transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies basic fill and encode label transform. Appropriate for tree-based models.
    """
    train, test = basic_fill(train, test)
    train, test = encode_label_transform(train, test)

    return train, test


def mlp_transform(train: pd.DataFrame, test: pd.DataFrame, targets: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies basic fill and one-hot encode transform. Appropriate for neural networks.
    """
    train, test = basic_fill(train, test)
    train, test = one_hot_encode_transform(train, test)
    train, test = normalize(train, test, excluded=targets)

    return train, test
