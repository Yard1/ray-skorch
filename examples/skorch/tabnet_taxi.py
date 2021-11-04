import argparse

import numpy as np
import pandas as pd
from skorch.callbacks.base import Callback

from torch import nn
import ray
import os

from ray.util.dask import ray_dask_get
import dask
import dask.array as da
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler

import ray.data
import torch
import torch.nn.functional as F

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet
from ray_sklearn.models.tabnet import TabNet

ray.data.set_progress_bars(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )

    args = parser.parse_args()

    target = "fare_amount"
    features = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "passenger_count"
    ]

    ray.init(address="auto")
    dask.config.set(scheduler=ray_dask_get)
    scaler = StandardScaler(copy=False)
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[features +
                                                            [target]].dropna())
    dataset = ray.data.from_dask(dask_df)

    print(dataset)
    print("dataset loaded")

    dtypes = [torch.float] * len(features)

    reg = RayTrainNeuralNet(
        TabNet,
        criterion=nn.MSELoss,
        num_workers=4,
        module__input_dim=len(features),
        module__output_dim=1,
        max_epochs=args.max_epochs,
        lr=0.02,
        batch_size=1024 / 4,
        verbose=5,
        device="cuda",
        iterator_train__feature_columns=features,
        iterator_valid__feature_columns=features,
        iterator_train__label_column_dtype=torch.float,
        iterator_valid__label_column_dtype=torch.float,
        iterator_train__feature_column_dtypes=dtypes,
        iterator_valid__feature_column_dtypes=dtypes,
        iterator_train__drop_last=True,
        iterator_valid__drop_last=True,
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(dataset, target)
    #print(reg.predict(X))

    print("Done!")
    print(reg.history_)
