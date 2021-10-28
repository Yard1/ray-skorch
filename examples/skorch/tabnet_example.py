import argparse

import numpy as np
from torch import nn

from ray_sklearn.models.tabnet import TabNet
from ray_sklearn.skorch_approach.base import RayTrainNeuralNet


def data_creator(size, a, b):
    X = np.arange(0, 10, 10 / size, dtype=np.float32).reshape((size, 1))
    y = a * X + b
    return (X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )

    args = parser.parse_args()

    X, y = data_creator(100, 2, 5)

    print("x", X.shape)
    print("y", y.shape)

    reg = RayTrainNeuralNet(
        TabNet,
        module__input_dim=1,
        module__output_dim=1,
        max_epochs=args.max_epochs,
        lr=0.1,
        criterion=nn.MSELoss,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    reg.fit(X, y)
    print(reg.predict(X))
