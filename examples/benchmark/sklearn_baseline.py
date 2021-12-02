import argparse
import os
import time

import dask
import dask.dataframe as dd
import ray
import torch
from dask_ml.preprocessing import StandardScaler
from ray.util.dask import ray_dask_get
from skorch.callbacks import GradientNormClipping
from torch import nn

from ray_sklearn.models.tabnet import TabNet
from ray_sklearn.skorch_approach.base import RayTrainNeuralNet
from ray_sklearn.skorch_approach.dataset import FixedSplit

ray.data.set_progress_bars(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If enabled, GPU will be used.",
    )
    parser.add_argument(
        "--address",
        type=str,
        required=False,
        help="The Ray address to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="If enabled, train on a smaller set of data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="The mini-batch size. Each worker will process "
             "batch-size/num-workers records at a time. Defaults to 1024.",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        required=False,
        help="The per-worker batch size. If set this will take precedence the batch-size argument.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If enabled, training data will be globally shuffled.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="The learning rate. Defaults to 0.02",
    )

    args = parser.parse_args()
    num_workers = args.num_workers
    use_gpu = args.use_gpu
    address = args.address
    num_epochs = args.num_epochs
    smoke_test = args.smoke_test
    batch_size = args.batch_size
    worker_batch_size = args.worker_batch_size
    shuffle = args.shuffle
    lr = args.lr

    target = "fare_amount"
    features = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "passenger_count"
    ]

    ray.init(address=address)
    dask.config.set(scheduler=ray_dask_get)
    scaler = StandardScaler(copy=False)
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[
            features + [target]].dropna().astype("float32"))
    if smoke_test:
        dask_df = dask_df.sample(frac=0.1)
    dataset = ray.data.from_dask(dask_df)

    print(dataset)
    print("dataset loaded")

    dtypes = [torch.float] * len(features)

    class _GradientNormClipping(GradientNormClipping):
        _on_all_ranks = True


    if worker_batch_size:
        print(f"Using worker batch size: {worker_batch_size}")
        train_worker_batch_size = worker_batch_size
    else:
        train_worker_batch_size = batch_size / num_workers
        print(f"Using global batch size: {batch_size}. "
              f"For {num_workers} workers the per worker batch size is {train_worker_batch_size}.")



    train_start = time.time()

    reg = RayTrainNeuralNet(
        TabNet,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        num_workers=num_workers,
        module__input_dim=len(features),
        module__output_dim=1,
        max_epochs=num_epochs,
        lr=lr,
        batch_size=train_worker_batch_size,
        train_split=FixedSplit(0.2, shuffle),
        verbose=5,
        # required for GPU
        callbacks=[_GradientNormClipping(1.0)],
        device="cuda" if use_gpu else "cpu",
        iterator_train__feature_columns=features,
        iterator_valid__feature_columns=features,
        iterator_train__label_column_dtype=torch.float,
        iterator_valid__label_column_dtype=torch.float,
        iterator_train__feature_column_dtypes=dtypes,
        iterator_valid__feature_column_dtypes=dtypes,
        iterator_train__drop_last=True,
        iterator_valid__drop_last=True,
    )
    reg.fit(dataset, target)

    train_end = time.time()
    train_time = train_end - train_start

    print(f"Training completed in {train_time} seconds.")


    print("Done!")
    print(reg.history_)
