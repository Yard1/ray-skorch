import argparse
import pandas as pd
import ray
from torch import nn

from ray.data import from_pandas

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet

from basic_example import (data_creator, RegressorModule,
                           RegressorModuleMultiInputList,
                           RegressorModuleMultiInputDict)

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

    args = parser.parse_args()
    ray.init(address=args.address)

    X, y = data_creator(2000, 20)

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"

    columns = list(X.columns)
    num_columns = X.shape[1]

    X["target"] = y

    device = "cuda" if args.use_gpu else "cpu"

    dataset = from_pandas(X)

    print("Running single input example")
    reg = RayTrainNeuralNet(
        RegressorModule,
        criterion=nn.MSELoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=1,
    )
    reg.fit(dataset, y.name)
    print("Single input example done!")

    print("Running multi input example")

    X["extra_column"] = X[columns[0]].copy()
    dataset = from_pandas(X)

    reg = RayTrainNeuralNet(
        RegressorModuleMultiInputList,
        criterion=nn.MSELoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=1,
        iterator_train__feature_columns=[columns, ["extra_column"]],
        iterator_valid__feature_columns=[columns, ["extra_column"]],
    )
    reg.fit(dataset, y.name)
    print("Multi input example with list done!")

    reg = RayTrainNeuralNet(
        RegressorModuleMultiInputDict,
        criterion=nn.MSELoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=1,
        iterator_train__feature_columns={
            "X": columns,
            "X_other": ["extra_column"]
        },
        iterator_valid__feature_columns={
            "X": columns,
            "X_other": ["extra_column"]
        },
    )
    reg.fit(dataset, y.name)
    print("Multi input example with dict done!")

    print("Done!")
