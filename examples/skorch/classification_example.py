import argparse
from typing import List

import numpy as np
import pandas as pd

import ray
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification

from ray_sklearn import RayTrainNeuralNet


def data_creator(rows, cols, n_classes=2):
    X_regr, y_regr = make_classification(
        rows,
        cols,
        n_informative=cols // 2,
        n_classes=n_classes,
        random_state=0)
    X_regr = X_regr.astype(np.float32)
    y_regr = y_regr.reshape(-1, 1)
    return (X_regr, y_regr)


class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_units=10,
            nonlin=F.relu,
    ):
        super().__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, output_dim)

    def forward(self, X: torch.Tensor, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


class ClassifierModuleMultiInputList(ClassifierModule):
    def forward(self, X_list: List[torch.Tensor], **kwargs):
        return super().forward(X_list[0], **kwargs)


class ClassifierModuleMultiInputDict(ClassifierModule):
    def forward(self, X: torch.Tensor, X_other: torch.Tensor, **kwargs):
        return super().forward(X, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        default=None,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of cpus to start ray with.")

    args = parser.parse_args()
    ray.init(address=args.address, num_cpus=args.num_cpus)

    X, y = data_creator(8000, 20)

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"

    num_columns = X.shape[1]
    device = "cuda" if args.use_gpu else "cpu"

    print("Running single input binary example")
    reg = RayTrainNeuralNet(
        ClassifierModule,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=2,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit(X, y)
    print("Single input binary example done!")

    print("Running multi input binary example")
    reg = RayTrainNeuralNet(
        ClassifierModuleMultiInputList,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=2,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit([X, X.copy()], y)
    print("Multi input binary example with list done!")

    reg = RayTrainNeuralNet(
        ClassifierModuleMultiInputDict,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=2,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit({"X": X, "X_other": X.copy()}, y)
    print("Multi input binary example with dict done!")

    X, y = data_creator(8000, 20, n_classes=10)

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"

    num_columns = X.shape[1]

    print("Running single input multilabel example")
    reg = RayTrainNeuralNet(
        ClassifierModule,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=10,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit(X, y)
    print("Single input multilabel example done!")

    print("Running multi input multilabel example")
    reg = RayTrainNeuralNet(
        ClassifierModuleMultiInputList,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=10,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit([X, X.copy()], y)
    print("Multi input multilabel example with list done!")

    reg = RayTrainNeuralNet(
        ClassifierModuleMultiInputDict,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=10,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit({"X": X, "X_other": X.copy()}, y)
    print("Multi input multilabel example with dict done!")

    print("Done!")
