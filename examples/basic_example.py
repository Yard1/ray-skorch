import argparse

import numpy as np

from ray_sklearn.tab_network import TabNetRegressor


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
        help="Sets the number of training epochs. Defaults to 5.")

    args = parser.parse_args()

    X, y = data_creator(2000, 5, 10)

    reg = TabNetRegressor()
    reg.fit(X, y, eval_set=[(X, y)], max_epochs=args.max_epochs)
