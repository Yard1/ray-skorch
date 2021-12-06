import argparse

import numpy as np
import pandas as pd

from torch import nn

from ray.data import from_pandas

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet

from basic_example import (data_creator, RegressorModule,
                           RegressorModuleMultiInputList,
                           RegressorModuleMultiInputDict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )

    args = parser.parse_args()

    X, y = data_creator(2000, 20)

    columns = [str(x) for x in range(20)]
    X = pd.DataFrame(X, columns=columns)
    y = pd.Series(y.ravel())
    y.name = "target"
    X["target"] = y

    dataset = from_pandas(X)

    print("Running single input example")
    reg = RayTrainNeuralNet(
        RegressorModule,
        criterion=nn.MSELoss,
        num_workers=4,
        max_epochs=args.max_epochs,
        lr=0.1,
        device="cpu",
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(dataset, y.name)
    print("Single input example done!")

    print("Running multi input example")

    X["extra_column"] = X["1"].copy()
    dataset = from_pandas(X)

    reg = RayTrainNeuralNet(
        RegressorModuleMultiInputList,
        criterion=nn.MSELoss,
        num_workers=4,
        max_epochs=args.max_epochs,
        lr=0.1,
        device="cpu",
        iterator_train__feature_columns=[columns, ["extra_column"]],
        iterator_valid__feature_columns=[columns, ["extra_column"]],
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(dataset, y.name)
    print("Multi input example with list done!")

    reg = RayTrainNeuralNet(
        RegressorModuleMultiInputDict,
        criterion=nn.MSELoss,
        num_workers=4,
        max_epochs=args.max_epochs,
        lr=0.1,
        device="cpu",
        iterator_train__feature_columns={
            "X": columns,
            "X_other": ["extra_column"]
        },
        iterator_valid__feature_columns={
            "X": columns,
            "X_other": ["extra_column"]
        },
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(dataset, y.name)
    print("Multi input example with dict done!")

    print("Done!")
