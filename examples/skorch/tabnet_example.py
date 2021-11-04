import argparse

import numpy as np
from torch import nn
from sklearn.datasets import make_regression

from ray_sklearn.models.tabnet import TabNet
from ray_sklearn.skorch_approach.base import RayTrainNeuralNet


def data_creator(rows, cols):
    X_regr, y_regr = make_regression(
        rows, cols, n_informative=cols // 2, random_state=0)
    X_regr = X_regr.astype(np.float32)
    y_regr = y_regr.astype(np.float32) / 100
    y_regr = y_regr.reshape(-1, 1)
    return (X_regr, y_regr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )

    args = parser.parse_args()

    X, y = data_creator(20000, 20)

    print("x", X.shape)
    print("y", y.shape)

    reg = RayTrainNeuralNet(
        TabNet,
        criterion=nn.MSELoss,
        num_workers=4,
        module__input_dim=20,
        module__output_dim=1,
        max_epochs=args.max_epochs,
        lr=0.1,
        # Shuffle training data on each epoch
        # iterator_train__shuffle=True,
    )
    reg.fit(X, y)
    #print(reg.predict(X))
