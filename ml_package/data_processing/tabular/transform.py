import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .metric import evaluate_log_transformation, get_numerical_columns, get_categorical_columns


def encode_label_transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms categorical columns into numerical columns using LabelEncoder.
    """
    # Separate numerical and categorical columns
    cat_cols = get_categorical_columns(train, test)

    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
    return train, test


def one_hot_encode_transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms categorical columns into numerical columns using one-hot encoding.
    Ensures that both train and test datasets have the same one-hot encoded columns.
    """
    # Identify categorical columns
    cat_cols = get_categorical_columns(train, test)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    train_encoded = ohe.fit_transform(train[cat_cols])
    test_encoded = ohe.transform(test[cat_cols])

    train_encoded_df = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out(cat_cols))
    test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(cat_cols))

    train_encoded_df.index = train.index
    test_encoded_df.index = test.index

    train = train.drop(columns=cat_cols).reset_index(drop=True)
    test = test.drop(columns=cat_cols).reset_index(drop=True)

    train_encoded_df = train_encoded_df.reset_index(drop=True)
    test_encoded_df = test_encoded_df.reset_index(drop=True)

    train_final = pd.concat([train, train_encoded_df], axis=1)
    test_final = pd.concat([test, test_encoded_df], axis=1)

    return train_final, test_final


def normalize(train: pd.DataFrame, test: pd.DataFrame, excluded: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalizes numerical columns to have a mean of 0 and a standard deviation of 1.
    """
    num_cols = get_numerical_columns(train, test)

    for col in num_cols:
        if excluded and col in excluded:
            continue
        mean = train[col].mean()
        # val_range instead of std to avoid high activations on sparse data
        val_range = train[col].max() - train[col].min()
        train[col] = (train[col] - mean) / val_range
        test[col] = (test[col] - mean) / val_range

    return train, test
