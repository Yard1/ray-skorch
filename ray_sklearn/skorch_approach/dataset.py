from typing import Union, Optional, Any

import numpy as np
import pandas as pd
from ray.data import Dataset, from_numpy, from_pandas
from ray.data.dataset_pipeline import DatasetPipeline
from skorch.dataset import Dataset as SkorchDataset
from skorch.utils import check_indexing, is_pandas_ndframe
import torch

from ray.data.impl.torch_iterable_dataset import TorchIterableDataset

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
            if y is not None and not isinstance(y, str):
                raise TypeError(
                    f"If X is a Ray Dataset, y must be a string, got {type(y)}"
                )
            self._len = X.count()
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


class WorkerDataset(SkorchDataset):
    pass
