import pandas as pd
from .metric import get_numerical_columns, get_categorical_columns


def basic_fill(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fills missing values. On numerical columns, it fills with the mean. On categorical columns, it fills with the most
    frequent value.
    """
    # Separate numerical and categorical columns
    num_cols = get_numerical_columns(train, test)
    cat_cols = get_categorical_columns(train, test)

    # Impute numerical columns with mean
    for col in num_cols:
        mean = train[col].mean()
        train[col] = train[col].fillna(mean)
        test[col] = test[col].fillna(mean)

    # Impute categorical columns with the most frequent value
    for col in cat_cols:
        mode = train[col].mode()[0]
        train[col] = train[col].fillna(mode)
        test[col] = test[col].fillna(mode)

    return train, test
