from typing import TYPE_CHECKING, Union
from ray.train.session import get_session
from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline

from skorch.utils import is_dataset

import torch

if TYPE_CHECKING:
    from ray_skorch.callbacks.skorch import (  # noqa: F401
        TrainSklearnCallback)
    from ray.train.callbacks import TrainingCallback  # noqa: F401


def is_in_train_session() -> bool:
    try:
        get_session()
        return True
    except ValueError:
        return False


def is_dataset_or_ray_dataset(x) -> bool:
    return is_dataset(x) or isinstance(x, (Dataset, DatasetPipeline))


def is_using_gpu(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_available()


def insert_before_substring(base_string: str, string_to_insert: str,
                            substring: str) -> str:
    idx = base_string.index(substring)
    return (base_string[:idx] + string_to_insert + base_string[idx:])


def add_callback_if_not_already_in(
        callback_name: str,
        callback: Union["TrainSklearnCallback", "TrainingCallback"],
        callback_list: list) -> bool:
    """Add a callback to the list if there isn't one with the same
    name or type.
    """
    if not any(name == callback_name or isinstance(callback, type(c))
               for name, c in callback_list):
        callback_list.append((callback_name, callback))
        return True
    return False
