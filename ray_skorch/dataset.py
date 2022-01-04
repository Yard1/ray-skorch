# Based on code from https://github.com/skorch-dev/skorch

# BSD 3-Clause License

# Copyright (c) 2017, Benjamin Bossan, Daniel Nouri, Marian Tietz
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Tuple, Union, Optional, Any, List

import numpy as np
import pandas as pd
import torch

from ray.data import Dataset, from_pandas
from ray.data.dataset_pipeline import DatasetPipeline
from skorch.dataset import Dataset as SkorchDataset
from skorch.utils import check_indexing, is_pandas_ndframe

LABEL_COLUMN = "_label"


def _pandas_get_name_or_column(x: Union[pd.Series, pd.DataFrame]) -> str:
    if x is None:
        return None
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


# The reason for having RayDataset and RayPipelineDataset separately
# is to allow for FixedSplit to work correctly. They may be unified
# in the future.


# TODO support lists and dicts
class RayDataset(SkorchDataset):
    """Wrapper allowing for validation and conversion to ``ray.data.Dataset``.

    The dataset will always yield a tuple of two values, the first
    from the data (``X``) and the second from the target (``y``).
    The data will always be a ``ray.data.Dataset``, and the target
    will always be a string, pertaining to the name of the target
    column in ``X``.

    :class:`.RayDataset` currently works with the following data types:

    * numpy arrays
    * pandas DataFrame or Series
    * a dictionary of the former two
    * a list/tuple of the former two
    * a ray.data.Dataset

    For ``ray.data.DatasetPipeline``s, please use :class:`.RayPipelineDataset`.

    Parameters
    ----------
    X : see above
      Everything pertaining to the input data.

    y : string or numpy array or pandas Series (default=None)
      If ``X`` is a ``ray.data`` object, this should be a string
      pertaining to the name of the label column inside ``X``.
      Otherwise, this should be a numpy array or pandas Series
      with the label data.

    length : int or None (default=None)
      Unused, left for compatibility.

    """

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

        if isinstance(X, DatasetPipeline):
            raise TypeError(
                "RayDataset doesn't support ray.data.DatasetPipeline. "
                "Please use RayPipelineDataset instead.")
        if isinstance(X, Dataset):
            self._init_dataset(X, y)
        else:
            super().__init__(X, y=y, length=length)
            # TODO ensure LABEL_COLUMN is not in X
            # TODO zip datasets instead of dataframes?
            if y is not None:
                if not is_pandas_ndframe(self.y):
                    self.y = pd.DataFrame(self.y, columns=[LABEL_COLUMN])
            else:
                self.y = None
            if isinstance(self.X, (list, tuple)):
                self.X = [
                    _convert_to_dataframe(x, i) for i, x in enumerate(self.X)
                ]
                self._X_multiple_input_columns = [x.columns for x in self.X]
                self.X = pd.concat(self.X, axis=1)
            elif isinstance(self.X, dict):
                self.X = {
                    k: _convert_to_dataframe(x, k)
                    for k, x in self.X.items()
                }
                self._X_multiple_input_columns = {
                    k: x.columns
                    for k, x in self.X.items()
                }
                self.X = list(self.X.values())
                self.X = pd.concat(self.X, axis=1)
            elif not is_pandas_ndframe(self.X):
                self.X = _convert_to_dataframe(self.X)
            if self.y is not None:
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
        if y and not isinstance(y, str):
            raise TypeError(
                f"If X is a Ray Dataset, y must be a string, got {type(y)}")
        self.X, self.y = self.convert(X, y)
        self.X_indexing = check_indexing(X)
        self.y_indexing = check_indexing(y)
        self.X_is_ndframe = is_pandas_ndframe(X)

    @property
    def X_multiple_input_columns(self):
        return getattr(self, "_X_multiple_input_columns", None)

    @X_multiple_input_columns.setter
    def X_multiple_input_columns(self, value):
        self._X_multiple_input_columns = value

    def get_params(self) -> Dict[str, Any]:
        return {"X_multiple_input_columns": self.X_multiple_input_columns}

    # TODO add validation
    def set_params(self, **params):
        for param_name, param in params.items():
            setattr(self, param_name, param)


class RayPipelineDataset(RayDataset):
    """Wrapper allowing for validation and conversion to
    ``ray.data.DatasetPipeline``.

    The dataset will always yield a tuple of two values, the first
    from the data (``X``) and the second from the target (``y``).
    The data will always be a ``ray.data.DatasetPipeline``, and the target
    will always be a string, pertaining to the name of the target
    column in ``X``.

    :class:`.RayPipelineDataset` currently works with the following data types:

    * numpy arrays
    * pandas DataFrame or Series
    * a dictionary of the former two
    * a list/tuple of the former two
    * a ray.data.Dataset
    * a ray.data.DatasetPipeline

    Parameters
    ----------
    X : see above
      Everything pertaining to the input data.

    y : see above or None (default=None)
      Everything pertaining to the target, if there is anything.

    length : int or None (default=None)
      Unused, left for compatibility.

    random_shuffle_each_window : bool (default=False)
        Whether to shuffle each window of the pipeline.

    """

    def __init__(
            self,
            X: Union[np.ndarray, Dataset],
            y: Optional[Union[np.ndarray, str]] = None,
            length=None,
            random_shuffle_each_window: bool = False,
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
    """Returns a :class:`.RayPipelineDataset` if ``X`` is a
    `ray.data.DatasetPipeline` and a :class:`.RayDataset` otherwise.

    If ``X`` is already a :class:`.RayDataset`, return ``X``.
    """
    if isinstance(X, RayDataset):
        return X
    if isinstance(X, DatasetPipeline):
        return RayPipelineDataset(X, y=y, **kwargs)
    return RayDataset(X, y=y, **kwargs)


class FixedSplit:
    """Class that performs the internal train/valid split on a
    ``ray.data.Dataset``.

    This class simply splits the dataset into two, similar to sklearn's
    ``train_test_split``.

    Parameters
    ----------
    valid_fraction : float (default=0.2)
      The proportion of the dataset to include in the validation split.

    shuffle : bool (default=True)
      Whether to shuffle the dataset before splitting.

    """

    def __init__(self, valid_fraction: float = 0.2, shuffle: bool = True):
        self.valid_fraction = valid_fraction
        self.shuffle = shuffle

    def __call__(self, dataset: RayDataset, y=None,
                 groups=None) -> Tuple[RayPipelineDataset, RayPipelineDataset]:
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
    """An iterator for returing batches from a :class:`.RayPipelineDataset`,
    similar to ``torch.utils.data.IterableDataset``.

    Internally, this class uses a modified version of the
    ``ray.data.Dataset.to_torch()`` method, with adjustments made to allow for
    not changing the shape of the label tensor and for returning of
    lists/dicts of feature tensors for multi-input modules. That method is,
    in turn, called on the items yielded by
    ``ray.data.Dataset.iter_epochs()``.

    Parameters
    ----------
    skorch_dataset : RayPipelineDataset
      The dataset to iterate over.

    batch_size : int
      How many samples per batch to yield at a time.

    feature_columns : list of names or list/dict of the former (default=None)
        The names of the columns to use as the features. If None, then use
        all columns except the label columns as the features. If a list/dict
        of lists is passed, the features will be split into multiple
        tensors to work with multi-input modules.

    label_column_dtype : torch.dtype (default=None)
        The torch dtype to use for the label column. If None, then
        automatically infer the dtype.

    feature_column_dtypes : list of torch.dtype or list/dict of the former
    (default=None)
        dtypes to use for the feature columns. The len(s) of this list must
        be equal to the len(s) of ``feature_columns``. If None,
        then automatically infer the dtype.

    prefetch_blocks : int (default=0)
        The number of blocks to prefetch ahead of the current block
        during the scan.

    drop_last : bool (default=False)
        Set to True to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If
        False and the size of dataset is not divisible by the batch
        size, then the last batch will be smaller.

    unsqueeze_label_tensor : bool (default=True)
        Whether the label tensor should be unsqueezed (reshaped to [n, 1])
        or not. In general regression loss functions such as ``MSELoss``
        require the tensor to be unsqueezed, and classification metrics
        such as ``CrossEntropyLoss`` require it to be 1D - for those, this
        argument should be set to False.

    """

    def __init__(
            self,
            skorch_dataset: RayPipelineDataset,
            batch_size: int,
            *,
            feature_columns: Optional[Union[List[str], Dict[str, List[str]],
                                            List[List[str]]]] = None,
            label_column_dtype: Optional["torch.dtype"] = None,
            feature_column_dtypes: Optional[Union[List["torch.dtype"], Dict[
                str, List["torch.dtype"]], List[List["torch.dtype"]]]] = None,
            prefetch_blocks: int = 0,
            drop_last: bool = False,
            unsqueeze_label_tensor: bool = True) -> None:
        self._validate_feature_columns(skorch_dataset, feature_columns,
                                       feature_column_dtypes)
        self.skorch_dataset = skorch_dataset
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.label_column_dtype = label_column_dtype
        self.feature_column_dtypes = feature_column_dtypes
        self.prefetch_blocks = prefetch_blocks
        self.drop_last = drop_last
        self.unsqueeze_label_tensor = unsqueeze_label_tensor
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
            drop_last: bool = False,
            unsqueeze_label_tensor: bool = False):
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
                if label_column:
                    label_vals = batch.pop(label_column).values
                    label_tensor = torch.as_tensor(
                        label_vals, dtype=label_column_dtype)
                    if unsqueeze_label_tensor:
                        label_tensor = label_tensor.view(-1, 1)
                else:
                    label_tensor = None

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
                    use_prefix = (self.skorch_dataset.X_multiple_input_columns
                                  and feature_columns)
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

    def __iter__(self) -> Tuple[Union["torch.Tensor", List[
            "torch.Tensor"], Dict[str, "torch.Tensor"]], "torch.Tensor"]:
        yield from self.to_torch(
            next(self._iterator),
            label_column=self.skorch_dataset.y,
            batch_size=self.batch_size,
            feature_columns=self.feature_columns,
            label_column_dtype=self.label_column_dtype,
            feature_column_dtypes=self.feature_column_dtypes,
            prefetch_blocks=self.prefetch_blocks,
            drop_last=self.drop_last,
            unsqueeze_label_tensor=self.unsqueeze_label_tensor)
