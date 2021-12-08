from ray.train.session import get_session
from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline

from skorch.utils import is_dataset

import torch


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


def add_callback_if_not_already_in(callback_name: str, callback,
                                   callback_list: list) -> bool:
    """Add a callback to the list if there isn't one with the same name or type."""
    if not any(name == callback_name or isinstance(callback, type(c))
               for name, c in callback_list):
        callback_list.append((callback_name, callback))
        return True
    return False
