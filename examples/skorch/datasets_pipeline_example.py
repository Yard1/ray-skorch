import argparse

import numpy as np
import pandas as pd

from torch import nn

from ray.data import from_pandas

from ray_sklearn.skorch_approach.base import RayTrainNeuralNet
from ray_sklearn.skorch_approach.dataset import RayDataset

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
    
    split = 0.2

    split_index = int(dataset.count() * split)

    train_dataset, validation_dataset = \
        dataset.random_shuffle().split_at_indices([split_index])

    train_dataset_pipeline = \
        train_dataset.repeat().random_shuffle_each_window()
    validation_dataset_pipeline = validation_dataset.repeat()


    reg = RayTrainNeuralNet(
        RegressorModule,
        criterion=nn.MSELoss,
        num_workers=4,
        max_epochs=args.max_epochs,
        lr=0.1,
        #train_split=None,
        # Shuffle training data on each epoch
        #iterator_train__shuffle=True,
    )
    reg.fit(train_dataset_pipeline, "target", X_val=validation_dataset_pipeline)
    #print(reg.predict(X))

    print("Done!")
