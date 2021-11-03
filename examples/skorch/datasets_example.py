import argparse

import numpy as np
import pandas as pd

from torch import nn

from ray.data import from_pandas

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet

from basic_example import data_creator, RegressorModule

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

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"
    X["target"] = y

    dataset = from_pandas(X)

    reg = RayTrainNeuralNet(
        RegressorModule,
        max_epochs=args.max_epochs,
        lr=0.1,
        criterion=nn.MSELoss,
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(dataset, "target")
    #print(reg.predict(X))

    print("Done!")
