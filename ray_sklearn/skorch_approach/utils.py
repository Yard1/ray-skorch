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


def is_using_gpu(device) -> bool:
    return device == "cuda" and torch.cuda.is_available()
