from typing import Dict, Union, Optional, Any, List

import numpy as np
import pandas as pd
import torch

from ray.data import Dataset, from_numpy, from_pandas
from ray.data.dataset_pipeline import DatasetPipeline
from skorch.dataset import Dataset as SkorchDataset
from skorch.utils import check_indexing, is_dataset, is_pandas_ndframe

LABEL_COLUMN = "_label"


def _pandas_get_name_or_column(x: Union[pd.Series, pd.DataFrame]) -> str:
    if hasattr(x, "name"):
        return x.name
    return x.columns[0]


def _convert_to_dataframe(x: Any, column_prefix: str = ""):
    if isinstance(x, (Dataset, DatasetPipeline)):
        raise TypeError("A dict/list of Ray Datasets is not supported. "
                        "Pass a single Ray Dataset as X, and then pass "
                        "a dict of [argument_name, columns] to "
                        "`iterator_train__feature_columns` "
                        "and `iterator_valid__feature_columns` in order "
                        "to pass multiple inputs to module.")
    if not is_pandas_ndframe(x):
        x = pd.DataFrame(x)
    else:
        x = x.copy()
    x.columns = [f"{column_prefix}{col}" for col in x.columns]
    return x


# TODO support lists and dicts
class RayDataset(SkorchDataset):
    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame, Dataset],
            y: Optional[Union[np.ndarray, pd.Series, str]] = None,
            length=None,
    ):

        # fit(X, y) -> X is a Ray Dataset, y is a string column
        # fit(X, y) -> X is a numpy matrix, y is a numpy vector

        self.X = X
        self.y = y

        if isinstance(X, Dataset):
            self._init_dataset(X, y)
        else:
            super().__init__(X, y=y, length=length)
            if y is not None:
                if not is_pandas_ndframe(self.y):
                    self.y = pd.DataFrame(self.y, columns=[LABEL_COLUMN])
            else:
                self.y = pd.DataFrame(
                    [False] * len(self.X), columns=[LABEL_COLUMN])
            if isinstance(self.X, (list, tuple)):
                self.X = [
                    _convert_to_dataframe(x, i) for i, x in enumerate(self.X)
                ]
                self.X_multiple_input_columns = [x.columns for x in self.X]
                self.X = pd.concat(self.X, axis=1)
            elif isinstance(self.X, dict):
                self.X = {
                    k: _convert_to_dataframe(x, k)
                    for k, x in self.X.items()
                }
                self.X_multiple_input_columns = {
                    k: x.columns
                    for k, x in self.X.items()
                }
                self.X = list(self.X.values())
                self.X = pd.concat(self.X, axis=1)
            elif not is_pandas_ndframe(self.X):
                self.X = _convert_to_dataframe(self.X)
                self.X_multiple_input_columns = None
            self.X = pd.concat((self.X, self.y), axis=1)
            self.X = from_pandas(self.X)
            self._init_dataset(self.X, _pandas_get_name_or_column(self.y))

    def __len__(self):
        return self._len

    def convert(self, X: Dataset, y=None):
        self._len = X.count()
        return X, y

    def _init_dataset(self, X: Dataset, y: str):
        if not isinstance(X, Dataset):
            raise TypeError(f"X must be a Ray Dataset, got {type(X)}")
        if not isinstance(y, str):
            raise TypeError(
                f"If X is a Ray Dataset, y must be a string, got {type(y)}")
        self.X, self.y = self.convert(X, y)
        self.X_indexing = check_indexing(X)
        self.y_indexing = check_indexing(y)
        self.X_is_ndframe = is_pandas_ndframe(X)
        if not hasattr(self, "X_multiple_input_columns"):
            self.X_multiple_input_columns = None

    def get_params(self) -> Dict[str, Any]:
        return {"X_multiple_input_columns": self.X_multiple_input_columns}

    # TODO add validation
    def set_params(self, **params):
        for param_name, param in params.items():
            setattr(self, param_name, param)


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
        # Make sure to carry over the params
        params = dataset.get_params()

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

        dataset_train.set_params(**params)
        dataset_valid.set_params(**params)
        return dataset_train, dataset_valid


class PipelineIterator:
    def __init__(
            self,
            skorch_dataset: RayPipelineDataset,
            batch_size: int,
            feature_columns: Optional[Union[List[str], Dict[str, List[str]],
                                            List[List[str]]]] = None,
            label_column_dtype: Optional["torch.dtype"] = None,
            feature_column_dtypes: Optional[Union[List["torch.dtype"], Dict[
                str, List["torch.dtype"]], List[List["torch.dtype"]]]] = None,
            prefetch_blocks: int = 0,
            drop_last: bool = False) -> None:
        self._validate_feature_columns(skorch_dataset, feature_columns,
                                       feature_column_dtypes)
        self.skorch_dataset = skorch_dataset
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.label_column_dtype = label_column_dtype
        self.feature_column_dtypes = feature_column_dtypes
        self.prefetch_blocks = prefetch_blocks
        self.drop_last = drop_last
        self._next_iter = None
        self._iterator = skorch_dataset.X.iter_epochs()

    def _validate_feature_columns(
            self, skorch_dataset: RayPipelineDataset,
            feature_columns: Optional[Union[List[str], Dict[str, List[str]],
                                            List[List[str]]]],
            feature_column_dtypes: Optional[Union[List["torch.dtype"], Dict[
                str, List["torch.dtype"]], List[List["torch.dtype"]]]]):
        # TODO add validation
        return

    def to_torch(
            self,
            dataset: Dataset,
            *,
            label_column: str,
            feature_columns: Optional[Union[List[str], Dict[str, List[str]],
                                            List[List[str]]]] = None,
            label_column_dtype: Optional["torch.dtype"] = None,
            feature_column_dtypes: Optional[Union[List["torch.dtype"], Dict[
                str, List["torch.dtype"]], List[List["torch.dtype"]]]] = None,
            batch_size: int = 1,
            prefetch_blocks: int = 0,
            drop_last: bool = False):
        """Copy of Dataset.to_torch with support for returning dicts/lists."""
        from ray.data.impl.torch_iterable_dataset import \
            TorchIterableDataset

        if feature_columns and feature_column_dtypes:
            if len(feature_columns) != len(feature_column_dtypes):
                raise ValueError("The lengths of `feature_columns` "
                                 f"({len(feature_columns)}) and "
                                 f"`feature_column_dtypes` ("
                                 f"{len(feature_column_dtypes)}) do not "
                                 "match!")

        def get_features_tensor(batch, feature_columns, feature_column_dtypes):
            feature_tensors = []
            if feature_columns:
                batch = batch[feature_columns]

            if feature_column_dtypes:
                dtypes = feature_column_dtypes
            else:
                dtypes = [None] * len(batch.columns)

            for col, dtype in zip(batch.columns, dtypes):
                col_vals = batch[col].values
                t = torch.as_tensor(col_vals, dtype=dtype)
                t = t.view(-1, 1)
                feature_tensors.append(t)

            return torch.cat(feature_tensors, dim=1)

        def make_generator():
            for batch in dataset.iter_batches(
                    batch_size=batch_size,
                    batch_format="pandas",
                    prefetch_blocks=prefetch_blocks,
                    drop_last=drop_last):
                label_vals = batch.pop(label_column).values
                label_tensor = torch.as_tensor(
                    label_vals, dtype=label_column_dtype)
                label_tensor = label_tensor.view(-1, 1)

                feature_columns_not_none = (
                    feature_columns
                    or self.skorch_dataset.X_multiple_input_columns)

                if feature_columns_not_none:
                    iterator = feature_columns_not_none.items() if isinstance(
                        feature_columns_not_none,
                        dict) else enumerate(feature_columns_not_none)

                    use_multi_input = (
                        self.skorch_dataset.X_multiple_input_columns
                        or isinstance(next(iter(iterator))[1], list))

                    # reset iterator
                    iterator = feature_columns_not_none.items() if isinstance(
                        feature_columns_not_none,
                        dict) else enumerate(feature_columns_not_none)
                else:
                    use_multi_input = False

                if use_multi_input:
                    use_prefix = self.skorch_dataset.X_multiple_input_columns and feature_columns
                    features_tensor = type(feature_columns_not_none)()
                    for k, v in iterator:
                        # Add prefix only if it's not already there
                        prefix = k if use_prefix else ""
                        feature_columns_element = [
                            f"{prefix}{col}"
                            for col in feature_columns_not_none[k]
                        ]

                        if feature_column_dtypes:
                            feature_column_dtypes_element = (
                                feature_column_dtypes[k])
                        else:
                            feature_column_dtypes_element = None

                        feature_tensor = get_features_tensor(
                            batch, feature_columns_element,
                            feature_column_dtypes_element)

                        if isinstance(features_tensor, dict):
                            features_tensor[k] = feature_tensor
                        else:
                            features_tensor.append(feature_tensor)
                else:
                    features_tensor = get_features_tensor(
                        batch, feature_columns, feature_column_dtypes)

                yield (features_tensor, label_tensor)

        return TorchIterableDataset(make_generator)

    def __iter__(self):
        yield from self.to_torch(
            next(self._iterator),
            label_column=self.skorch_dataset.y,
            batch_size=self.batch_size,
            feature_columns=self.feature_columns,
            label_column_dtype=self.label_column_dtype,
            feature_column_dtypes=self.feature_column_dtypes,
            prefetch_blocks=self.prefetch_blocks,
            drop_last=self.drop_last)
