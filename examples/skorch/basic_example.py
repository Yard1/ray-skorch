import argparse

import numpy as np
import pandas as pd

from torch import nn
import torch.nn.functional as F
from sklearn.datasets import make_regression

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet


def data_creator(rows, cols):
    X_regr, y_regr = make_regression(
        rows, cols, n_informative=cols // 2, random_state=0)
    X_regr = X_regr.astype(np.float32)
    y_regr = y_regr.astype(np.float32) / 100
    y_regr = y_regr.reshape(-1, 1)
    return (X_regr, y_regr)


class RegressorModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

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

    X, y = data_creator(2000, 20)

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"

    reg = RayTrainNeuralNet(
        RegressorModule,
        max_epochs=args.max_epochs,
        lr=0.1,
        criterion=nn.MSELoss,
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(X, y)
    #print(reg.predict(X))

    print("Done!")