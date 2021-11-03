from typing import Union, Optional, Any

import numpy as np
import pandas as pd
from ray.data import Dataset, from_numpy, from_pandas
from ray.data.dataset_pipeline import DatasetPipeline
from skorch.dataset import Dataset as SkorchDataset
from skorch.utils import check_indexing, is_dataset, is_pandas_ndframe

LABEL_COLUMN = "label"


def _pandas_get_name_or_column(x: Union[pd.Series, pd.DataFrame]) -> str:
    if hasattr(x, "name"):
        return x.name
    return x.columns[0]


class RayDataset(SkorchDataset):
    def __init__(
            self,
            X: Union[np.ndarray, Dataset],
            y: Optional[Union[np.ndarray, str]] = None,
            length=None,
    ):

        # fit(X, y) -> X is a Ray Dataset, y is a string column
        # fit(X, y) -> X is a numpy matrix, y is a numpy vector

        self.X = X
        self.y = y

        if isinstance(X, Dataset):
            if not isinstance(y, str):
                raise TypeError(
                    f"If X is a Ray Dataset, y must be a string, got {type(y)}"
                )
            self.X, self.y = self.convert(self.X, self.y)
            self.X_indexing = check_indexing(X)
            self.y_indexing = check_indexing(y)
            self.X_is_ndframe = is_pandas_ndframe(X)
        else:
            super().__init__(X, y=y, length=length)
            if y is not None:
                if not is_pandas_ndframe(self.y):
                    self.y = pd.DataFrame(self.y, columns=[LABEL_COLUMN])
            else:
                self.y = pd.DataFrame(
                    [False] * len(self.X), columns=[LABEL_COLUMN])
            if not is_pandas_ndframe(self.X):
                self.X = pd.DataFrame(self.X)
            self.X = from_pandas(pd.concat((self.X, self.y), axis=1))
            self.__init__(
                self.X, y=_pandas_get_name_or_column(self.y), length=length)

    def __len__(self):
        return self._len

    def convert(self, X: Dataset, y=None):
        self._len = X.count()
        return X, y


class RayPipelineDataset(RayDataset):
    def __init__(
            self,
            X: Union[np.ndarray, Dataset],
            y: Optional[Union[np.ndarray, str]] = None,
            length=None,
            random_shuffle_each_window: bool = True,
    ):
        self.random_shuffle_each_window = random_shuffle_each_window
        if isinstance(X, DatasetPipeline):
            self.X = X
            self.y = y
            self.X_indexing = check_indexing(X)
            self.y_indexing = check_indexing(y)
            self.X_is_ndframe = is_pandas_ndframe(X)
        else:
            super().__init__(X, y=y, length=length)

    def convert(self, X: Dataset, y=None):
        self._len = X.count()
        X_pipeline = X.repeat()
        if self.random_shuffle_each_window:
            X_pipeline = X_pipeline.random_shuffle_each_window()
        return X_pipeline, y


def dataset_factory(X, y=None,
                    **kwargs) -> Union[RayPipelineDataset, RayDataset]:
    if is_dataset(X):
        return X
    if isinstance(X, DatasetPipeline):
        return RayPipelineDataset(X, y=y, **kwargs)
    return RayDataset(X, y=y, **kwargs)


class FixedSplit:
    def __init__(self, valid_fraction: float = 0.2, shuffle: bool = True):
        self.valid_fraction = valid_fraction
        self.shuffle = shuffle

    def __call__(self, dataset: RayDataset, y=None, groups=None) -> Any:
        X = dataset.X

        if isinstance(X, DatasetPipeline):
            raise TypeError(
                "Automatic train-validation split is not supported if X is a "
                "DatasetPipeline. Pass a separate DatasetPipeline as the "
                "`X_val` argument instead.")
        elif not isinstance(X, Dataset):
            raise TypeError(f"X must be a Dataset, got {type(X)}.")

        split_index = int(X.count() * (1 - self.valid_fraction))

        if self.shuffle:
            X = X.random_shuffle()

        X_train, X_valid = X.split_at_indices([split_index])

        dataset_train = RayPipelineDataset(
            X_train, y=dataset.y, random_shuffle_each_window=self.shuffle)
        dataset_valid = RayPipelineDataset(
            X_valid, y=dataset.y, random_shuffle_each_window=False)

        return dataset_train, dataset_valid


class PipelineIterator:
    def __init__(self, dataset: RayPipelineDataset, batch_size) -> None:
        self.dataset = dataset
        self.iterator = dataset.X.iter_epochs()
        self.batch_size = batch_size
        self.next_iter = None

    def __iter__(self):
        self.next_iter = iter(
            next(self.iterator).to_torch(
                label_column=self.dataset.y, batch_size=self.batch_size))
        return self

    def __next__(self):
        return next(self.next_iter)
