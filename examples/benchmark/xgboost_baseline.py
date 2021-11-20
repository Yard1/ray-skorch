import argparse
import os
import time

import dask
import dask.dataframe as dd
import ray.data
from dask_ml.preprocessing import StandardScaler
from ray.util.dask import ray_dask_get
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
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If enabled, GPU will be used.",
    )
    parser.add_argument(
        "--address",
        type=str,
        required=False,
        help="The Ray address to use.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="If enabled, train on a smaller set of data.",
    )

    args = parser.parse_args()
    num_workers = args.num_workers
    use_gpu = args.use_gpu
    address = args.address
    smoke_test = args.smoke_test

    target = "fare_amount"
    features = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "passenger_count"
    ]

    ray.init(address=address)
    dask.config.set(scheduler=ray_dask_get)
    scaler = StandardScaler(copy=False)
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[
            features + [target]].dropna().astype("float32"))
    dataset = ray.data.from_dask(dask_df)
    if smoke_test:
        dataset = dataset.limit(2000).repartition(128)

    print(dataset)
    print("dataset loaded")

    train_set = RayDMatrix(dataset, target)

    evals_result = {}
    # Set XGBoost config.
    xgboost_params = {
        "tree_method": "approx",
        "objective": "reg:squarederror",
        "eval_metric": ["rmse"],
    }

    train_start = time.time()

    # Train the classifier
    bst = train(
        params=xgboost_params,
        dtrain=train_set,
        evals=[(train_set, "train")],
        evals_result=evals_result,
        ray_params=RayParams(
            max_actor_restarts=0,
            gpus_per_actor=int(use_gpu),
            cpus_per_actor=2,
            num_actors=num_workers),
        verbose_eval=False,
        num_boost_round=10)

    train_end = time.time()
    train_time = train_end - train_start

    print(f"Training completed in {train_time} seconds.")


    model_path = "ray_datasets.xgb"
    bst.save_model(model_path)
    rmse = evals_result["train"]["rmse"][-1]
    print(evals_result["train"])
    print("Final training error (RMSE): {:.4f}".format(rmse))
    print("Final training error (MSE): {:.4f}".format(rmse**2))