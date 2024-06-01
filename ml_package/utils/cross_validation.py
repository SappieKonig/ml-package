import random
from typing import Generator, Tuple

from ml_package.types import MassIndex
from .select import select_on_index


def cross_validation(to_split: list[MassIndex], n_folds: int = 5, shuffle: bool = True) -> Generator[Tuple[MassIndex, MassIndex, MassIndex, MassIndex], None, None]:
    n_samples = len(to_split[0])
    assert all(len(data) == n_samples for data in to_split), "All data must have the same length"
    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)

    for i in range(n_folds):
        start = i * n_samples // n_folds
        end = (i + 1) * n_samples // n_folds
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        yield [(select_on_index(data, train_indices), select_on_index(data, val_indices)) for data in to_split]
