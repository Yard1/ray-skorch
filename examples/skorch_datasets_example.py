import argparse

import pandas as pd

from torch import nn

import ray
import ray.data
from train_sklearn import RayTrainNeuralNet
from train_sklearn.dataset import RayDataset

from basic_example import data_creator, RegressorModule

ray.data.set_progress_bars(False)

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
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of cpus to start ray with.")

    args = parser.parse_args()
    ray.init(address=args.address, num_cpus=args.num_cpus)

    X, y = data_creator(2000, 20)

    X = pd.DataFrame(X)
    y = pd.Series(y.ravel())
    y.name = "target"

    num_columns = X.shape[1]
    device = "cuda" if args.use_gpu else "cpu"

    dataset = RayDataset(X, y)

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
    reg.fit(dataset, "target")

    print("Done!")
