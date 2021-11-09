import argparse

import numpy as np
import pandas as pd

from torch import nn
import ray
import os

from ray.util.dask import ray_dask_get
import dask
import dask.array as da
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler

import ray.data
from xgboost_ray import RayDMatrix, train, RayParams

ray.data.set_progress_bars(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers.",
    )

    args = parser.parse_args()

    target = "fare_amount"
    features = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "passenger_count"
    ]

    ray.init(address="auto")
    dask.config.set(scheduler=ray_dask_get)
    scaler = StandardScaler(copy=False)
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[
            features + [target]].dropna().astype("float32"))
    dataset = ray.data.from_dask(dask_df)  #.limit(1000000).repartition(128)

    print(dataset)
    print("dataset loaded")


    train_set = RayDMatrix(dataset, target)

    evals_result = {}
    # Set XGBoost config.
    xgboost_params = {
        "tree_method": "approx",
        "objective": "reg:squarederror",
        "eval_metric": ["logloss", "error"],
    }

    # Train the classifier
    bst = train(
        params=xgboost_params,
        dtrain=train_set,
        evals=[(train_set, "train")],
        evals_result=evals_result,
        ray_params=RayParams(
            max_actor_restarts=0,
            gpus_per_actor=1,
            cpus_per_actor=2,
            num_actors=args.num_workers),
        verbose_eval=False,
        num_boost_round=10)

    model_path = "ray_datasets.xgb"
    bst.save_model(model_path)
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))
    print(evals_result["train"])