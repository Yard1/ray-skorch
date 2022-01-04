import os
import argparse
import torch
from torch import nn

from ray.data import read_csv
import pyarrow.csv
from ray_skorch import RayTrainNeuralNet
from pytorch_tabnet.tab_network import TabNet

FILENAME_CSV = "HIGGS.csv.gz"


def download_higgs(target_file):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
          "00280/HIGGS.csv.gz"

    try:
        import urllib.request
    except ImportError as e:
        raise ValueError(
            f"Automatic downloading of the HIGGS dataset requires `urllib`."
            f"\nFIX THIS by running `pip install urllib` or manually "
            f"downloading the dataset from {url}.") from e

    print(f"Downloading HIGGS dataset to {target_file}")
    urllib.request.urlretrieve(url, target_file)
    return os.path.exists(target_file)


def main(args):
    # Example adapted from this blog post:
    # https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7
    # This uses the HIGGS dataset. Download here:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

    if not os.path.exists(FILENAME_CSV):
        assert download_higgs(FILENAME_CSV), \
            "Downloading of HIGGS dataset failed."
        print("HIGGS dataset downloaded.")
    else:
        print("HIGGS dataset found locally.")

    colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]
    num_feature_columns = len(colnames) - 1

    dataset = read_csv(
        os.path.abspath(FILENAME_CSV),
        read_options=pyarrow.csv.ReadOptions(
            use_threads=False, column_names=colnames))

    reg = RayTrainNeuralNet(
        TabNet,
        criterion=nn.CrossEntropyLoss,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=0.1,
        batch_size=8192 // args.num_workers,
        device="cuda" if args.use_gpu else "cpu",
        module__input_dim=num_feature_columns,
        module__output_dim=2,
        iterator_train__label_column_dtype=torch.long,
        iterator_valid__label_column_dtype=torch.long,
        iterator_train__feature_column_dtypes=[torch.float
                                               ] * num_feature_columns,
        iterator_valid__feature_column_dtypes=[torch.float] *
        num_feature_columns,
        # the following two arguments are required
        # to ensure that a 1-D tensor is passed to
        # the loss function, preventing an exception
        iterator_train__unsqueeze_label_tensor=False,
        iterator_valid__unsqueeze_label_tensor=False,
    )
    reg.fit(dataset, "label")


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

    main(args=parser.parse_args())
