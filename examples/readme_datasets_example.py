import argparse
import ray

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from torch import nn
from ray.data import from_pandas

from ray_skorch import RayTrainNeuralNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        default=None,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of cpus to start ray with.")

    args = parser.parse_args()
    ray.init(address=args.address, num_cpus=args.num_cpus)

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = pd.DataFrame(X.astype(np.float32))
    y = pd.Series(y.astype(np.int64))

    X_pred = X.copy()
    X["target"] = y

    X = from_pandas(X)
    # ensure no target column is in data for prediction
    X_pred = from_pandas(X_pred)

    class MyModule(nn.Module):
        def __init__(self, num_units=10, nonlin=nn.ReLU()):
            super(MyModule, self).__init__()

            self.dense0 = nn.Linear(20, num_units)
            self.nonlin = nonlin
            self.dropout = nn.Dropout(0.5)
            self.dense1 = nn.Linear(num_units, num_units)
            self.output = nn.Linear(num_units, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            X = self.nonlin(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    net = RayTrainNeuralNet(
        MyModule,
        num_workers=2,  # the only new mandatory argument
        criterion=nn.CrossEntropyLoss,
        max_epochs=10,
        lr=0.1,
        # required for classification loss funcs
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )

    net.fit(X, "target")

    # predict_proba returns a ray.data.Dataset
    y_proba = net.predict_proba(X_pred).to_pandas()
    print(y_proba)
