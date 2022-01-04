# train-sklearn

[skorch](https://github.com/skorch-dev/skorch)-based wrapper for [Ray Train](https://docs.ray.io/en/latest/train/train.html). Experimental!

1. Run `pip install -e .` to install necessary packages
2. Upon push, run `./format.sh` to make sure lint changes are applied appropriately.
3. The current working examples can be found in `examples`.

> :warning: `RayTrainNeuralNet` and the rest of this package are experimental and not production ready. In particular, validation and error handling may be spotty. If you encounter any problems or have any suggestions please open an issue on GitHub.

## Basic example

### With numpy/pandas

```python
import numpy as np
from sklearn.datasets import make_classification
from torch import nn

from train_sklearn import RayTrainNeuralNet

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)


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
    num_workers=2,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    lr=0.1,
    # required for classification loss funcs
    iterator_train__unsqueeze_label_tensor=False,
    iterator_valid__unsqueeze_label_tensor=False,
)

net.fit(X, y)

# predict_proba returns a ray.data.Dataset
y_proba = net.predict_proba(X).to_pandas()
print(y_proba)
```

### With Ray Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from torch import nn
from ray.data import from_pandas

from train_sklearn import RayTrainNeuralNet

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
    num_workers=2,
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
```