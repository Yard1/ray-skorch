import argparse
import pandas as pd
import ray
from torch import nn

from ray.data import from_pandas

from ray_sklearn import RayTrainNeuralNet

from basic_example import data_creator, RegressorModule

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

    columns = list(X.columns)
    num_columns = X.shape[1]

    X["target"] = y

    device = "cuda" if args.use_gpu else "cpu"

    dataset = from_pandas(X)

    assert args.epochs > 1
    first_epochs = args.epochs // 2
    second_epochs = args.epochs - first_epochs

    reg = RayTrainNeuralNet(
        RegressorModule,
        criterion=nn.MSELoss,
        num_workers=args.num_workers,
        max_epochs=first_epochs,
        lr=0.1,
        device=device,
        module__input_dim=num_columns,
        module__output_dim=1,
        save_checkpoints=True,
    )
    reg.fit(dataset, "target")
    assert reg.latest_checkpoint_
    reg.set_params(max_epochs=second_epochs)
    reg.fit(dataset, "target", checkpoint=reg.latest_checkpoint_)
    print(reg.history)
    assert reg.history[-1]["epoch"] == args.epochs

    print("Done!")
