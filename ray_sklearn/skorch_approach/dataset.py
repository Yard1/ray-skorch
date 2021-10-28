from typing import Union, Optional

import numpy as np
import pandas as pd
from ray.data import Dataset, from_numpy, from_pandas
from ray.data.dataset_pipeline import DatasetPipeline
from skorch.dataset import Dataset as SkorchDataset
import torch

from ray.data.impl.torch_iterable_dataset import TorchIterableDataset

LABEL_COLUMN = "label"


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
            self._len = X.count()
        else:
            super().__init__(X, y=y, length=length)
            self.y = pd.DataFrame(self.y, columns=[LABEL_COLUMN])
            self.X = pd.DataFrame(X)
            self.X = from_pandas(pd.concat((self.X, self.y), axis=1))
            self.__init__(self.X, y=LABEL_COLUMN, length=length)

        def __len__(self):
            return self._len


class WorkerDataset(SkorchDataset):
    pass
