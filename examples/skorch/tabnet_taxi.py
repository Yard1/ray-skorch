import argparse

import ray
import ray.data
import os

from ray.util.dask import ray_dask_get
import dask
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F

from skorch.callbacks import GradientNormClipping
from ray_sklearn import RayTrainNeuralNet
from ray_sklearn.models.tabnet import TabNet

ray.data.set_progress_bars(False)


class RegressorModule(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_units=10,
            nonlin=F.relu,
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, output_dim)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


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
    print("loading dataset")
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[
            features + [target]].dropna().astype("float32"))
    dataset = ray.data.from_dask(dask_df)

    print(dataset)
    print("dataset loaded")

    dtypes = [torch.float] * len(features)

    class _GradientNormClipping(GradientNormClipping):
        _on_all_ranks = True

    reg = RayTrainNeuralNet(
        TabNet,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        num_workers=4,
        module__input_dim=len(features),
        module__output_dim=1,
        max_epochs=args.max_epochs,
        lr=0.02,
        batch_size=1024 / 4,
        verbose=5,
        callbacks=[_GradientNormClipping(1.0)],
        device="cuda",
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

    print("Done!")
    print(reg.history_)
