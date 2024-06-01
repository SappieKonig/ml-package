import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def evaluate_log_transformation(column: pd.Series) -> bool:
    """
    Evaluates whether a logarithm transformation is appropriate for the given numerical column.
    Currently only properly handles decent sized series of positive large (>> 1) numbers.
    """
    # Calculate skewness and kurtosis
    original_skewness = skew(column)
    original_kurtosis = kurtosis(column)

    # Log transform (add a small constant to avoid log(0))
    log_transformed = np.log1p(column)

    # Calculate skewness and kurtosis after transformation
    transformed_skewness = skew(log_transformed)
    transformed_kurtosis = kurtosis(log_transformed)

    print(f"Original skewness: {original_skewness}")
    print(f"Transformed skewness: {transformed_skewness}")
    print(f"Original kurtosis: {original_kurtosis}")
    print(f"Transformed kurtosis: {transformed_kurtosis}")

    metric1 = abs(original_skewness) > abs(transformed_skewness)
    metric2 = original_kurtosis > transformed_kurtosis

    return metric1 and metric2


def get_categorical_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """
    Gets the categorical columns present in both the training and test datasets.
    """
    cat_cols_train = train.select_dtypes(include=['object', 'category']).columns
    cat_cols_test = test.select_dtypes(include=['object', 'category']).columns

    common_cat_cols = list(set(cat_cols_train) & set(cat_cols_test))

    return common_cat_cols


def get_numerical_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """
    Gets the numerical columns present in both the training and test datasets.
    """
    num_cols_train = train.select_dtypes(include=['number']).columns
    num_cols_test = test.select_dtypes(include=['number']).columns

    common_num_cols = list(set(num_cols_train) & set(num_cols_test))

    return common_num_cols
