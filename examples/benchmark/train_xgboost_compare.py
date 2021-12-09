import argparse
import json
import numbers
import os
import time
from typing import List, Dict

import dask
import dask.dataframe as dd
import numpy as np
import ray
import torch
import xgboost_ray
from dask_ml.preprocessing import StandardScaler
from ray import train
from ray.train import Trainer, CheckpointStrategy
from ray.train.callbacks import JsonLoggerCallback
from ray.util.dask import ray_dask_get
from ray.util.ml_utils.json import SafeFallbackEncoder
from torch import nn
from xgboost_ray import RayDMatrix, RayParams

from sklearn.metrics import mean_squared_error

from ray_sklearn.models.tabnet import TabNet

ray.data.set_progress_bars(False)

def max_and_argmax(val):
    return np.max(val), np.argmax(val)


def min_and_argmin(val):
    return np.min(val), np.argmin(val)


DEFAULT_AGGREGATE_FUNC = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "max": max_and_argmax,
    "min": min_and_argmin
}

DEFAULT_KEYS_TO_IGNORE = {
    "epoch", "_timestamp", "_training_iteration", "train_batch_size",
    "valid_batch_size", "lr"
}

class AggregateLogCallback(JsonLoggerCallback):
    def handle_result(self, results: List[Dict], **info):
        results_dict = {idx: val for idx, val in enumerate(results)}

        aggregate_results = {}
        for key, value in results_dict[0].items():
            if key in DEFAULT_KEYS_TO_IGNORE:
                aggregate_results[key] = value
            elif isinstance(value, numbers.Number):
                aggregate_key = [
                    result[key] for result in results if key in result
                ]

                aggregate = {}
                for func_key, func in DEFAULT_AGGREGATE_FUNC.items():
                    aggregate[func_key] = func(aggregate_key)
                aggregate_results[key] = aggregate


        final_results = {}
        final_results["raw"] = results_dict
        final_results["aggregated"] = aggregate_results

        with open(self._log_path, "r+") as f:
            loaded_results = json.load(f)
            f.seek(0)
            json.dump(
                loaded_results + [final_results], f, cls=SafeFallbackEncoder)

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
        "--num-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of data to train on",
    )
    # parser.add_argument(
    #     "--num_rows",
    #     type=float,
    #     default=1.0,
    #     help="Fraction of data to train on",
    # )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="The mini-batch size. Each worker will process "
             "batch-size/num-workers records at a time. Defaults to 1024.",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        required=False,
        help="The per-worker batch size. If set this will take precedence the batch-size argument.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If enabled, training data will be globally shuffled.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="The learning rate. Defaults to 0.02",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="If enabled, debug logs will be printed.",
    )

    args = parser.parse_args()
    num_workers = args.num_workers
    use_gpu = args.use_gpu
    address = args.address
    num_epochs = args.num_epochs
    fraction = args.fraction
    batch_size = args.batch_size
    worker_batch_size = args.worker_batch_size
    shuffle = args.shuffle
    lr = args.lr
    debug = args.debug

    target = "fare_amount"
    features = [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "passenger_count"
    ]

    ray.init(address=address)

    data_path = "./data"
    if os.path.exists(data_path):
        dataset = ray.data.read_csv(data_path)
    else:
        dask.config.set(scheduler=ray_dask_get)
        scaler = StandardScaler(copy=False)
        dask_df = scaler.fit_transform(
            dd.read_csv(os.path.expanduser("~/data/train.csv"))[
                features + [target]].dropna().astype("float32"))
        dask_df = dask_df.sample(frac=fraction)
        dataset = ray.data.from_dask(dask_df)
        dataset.write_csv(data_path)

    print(dataset)
    print("dataset loaded")

    dtypes = [torch.float] * len(features)

    if shuffle:
        dataset = dataset.random_shuffle()


    val_split = 0.1
    test_split = 0.2
    train_split = 1 - val_split - test_split

    num_records = dataset.count()
    train_val_split = int(num_records * train_split)
    val_test_split = int(num_records * (train_split + test_split))
    train_dataset, validation_dataset, test_dataset = dataset.split_at_indices(
        [train_val_split, val_test_split])

    train_dataset_pipeline = train_dataset.repeat(num_epochs)
    if shuffle:
        train_dataset_pipeline = train_dataset_pipeline.random_shuffle_each_window()
    validation_dataset_pipeline = validation_dataset.repeat(num_epochs)

    datasets = {
        "train_dataset_pipeline": train_dataset_pipeline,
        "validation_dataset_pipeline": validation_dataset_pipeline
    }


    if worker_batch_size:
        print(f"Using worker batch size: {worker_batch_size}")
        train_worker_batch_size = worker_batch_size
    else:
        train_worker_batch_size = batch_size / num_workers
        print(f"Using global batch size: {batch_size}. "
              f"For {num_workers} workers the per worker batch size is {train_worker_batch_size}.")


    def train_func(config):
        print(datasets)
        train_dataset_pipeline = train.get_dataset_shard(
            "train_dataset_pipeline")
        validation_dataset_pipeline = train.get_dataset_shard(
            "validation_dataset_pipeline")
        train_dataset_iterator = train_dataset_pipeline.iter_epochs()
        validation_dataset_iterator = validation_dataset_pipeline.iter_epochs()

        model = TabNet(input_dim=len(features), output_dim=1)
        model = train.torch.prepare_model(model, ddp_kwargs={
            "find_unused_parameters": True})
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10, threshold=1e-5)

        device = train.torch.get_device()

        results = []
        for i in range(num_epochs):
            train_dataset = next(train_dataset_iterator)
            validation_dataset = next(validation_dataset_iterator)
            train_torch_dataset = train_dataset.to_torch(
                label_column=target, label_column_dtype=torch.float, feature_columns=features, feature_column_dtypes=dtypes, batch_size=train_worker_batch_size)
            validation_torch_dataset = validation_dataset.to_torch(
                label_column=target, label_column_dtype=torch.float, feature_columns=features, feature_column_dtypes=dtypes, batch_size=train_worker_batch_size)

            last_time = time.time()

            model.train()
            train_train_loss = 0
            train_num_rows = 0
            for batch_idx, (X, y) in enumerate(train_torch_dataset):
                if debug and batch_idx % 1000 == 0:
                    curr_time = time.time()
                    time_since_last = curr_time - last_time
                    last_time = curr_time
                    print(f"Train epoch: [{i}], batch[{batch_idx}], time[{time_since_last}]")



                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                count = len(X)
                train_train_loss += count * loss.item()
                train_num_rows += count

            train_loss = train_train_loss / train_num_rows  # TODO: should this be num batches or num rows?
            print(f"Train epoch: [{i}], mean square error:[{train_loss}]")

            last_time = time.time()

            model.eval()
            val_total_loss = 0
            val_num_rows = 0
            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(validation_torch_dataset):
                    if debug and batch_idx % 1000 == 0:
                        curr_time = time.time()
                        time_since_last = curr_time - last_time
                        last_time = curr_time
                        print(f"Validation epoch: [{i}], batch[{batch_idx}], time[{time_since_last}]")
                    X = X.to(device)
                    y = y.to(device)
                    count = len(X)
                    pred = model(X)
                    val_total_loss += count * criterion(pred, y).item()
                    val_num_rows += count
            val_loss = val_total_loss / val_num_rows  # TODO: should this be num batches or num rows?
            print(f"Validation epoch: [{i}], mean square error:[{val_loss}]")

            scheduler.step(train_loss)
            curr_lr = [ group['lr'] for group in optimizer.param_groups ]

            train.report(train_mse=train_loss, val_mse=val_loss, lr=curr_lr)


            state_dict = model.state_dict()
            from torch.nn.modules.utils import \
                consume_prefix_in_state_dict_if_present
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
            train.save_checkpoint(val_mse=val_loss, model_state_dict=state_dict)
            results.append(val_loss)

        return results




    test_df = test_dataset.to_pandas()

    X_test = test_df[features]
    y_true = test_df[target]


    ############
    # xgboost
    ############

    train_set = RayDMatrix(train_dataset, target)

    evals_result = {}
    # Set XGBoost config.
    xgboost_params = {
        "tree_method": "approx",
        "objective": "reg:squarederror",
        "eval_metric": ["rmse"],
    }

    train_start = time.time()
    ray_params = RayParams(
        max_actor_restarts=0,
        gpus_per_actor=int(use_gpu),
        cpus_per_actor=2,
        num_actors=num_workers)


    # Train the classifier
    bst = xgboost_ray.train(
        params=xgboost_params,
        dtrain=train_set,
        evals=[(train_set, "train")],
        evals_result=evals_result,
        ray_params=ray_params,
        verbose_eval=False,
        num_boost_round=10)

    test_set = RayDMatrix(test_dataset, target)
    pred = xgboost_ray.predict(bst, test_set, ray_params=ray_params)

    xgb_loss = mean_squared_error(y_true, pred)
    print(f"xgboost test loss: {xgb_loss}")

    # import sys
    # sys.exit()

    # train_end = time.time()
    # train_time = train_end - train_start
    #
    # print(f"Training completed in {train_time} seconds.")



    ############
    # train
    ############





    train_start = time.time()

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    results = trainer.run(train_func, dataset=datasets, callbacks=[AggregateLogCallback()], checkpoint_strategy=CheckpointStrategy(checkpoint_score_attribute="val_mse", checkpoint_score_order="min"))
    trainer.shutdown()


    train_end = time.time()
    train_time = train_end - train_start

    print(f"Training completed in {train_time} seconds.")

    bcp = trainer.best_checkpoint_path
    print(f"bcp: {bcp}")
    with bcp.open("rb") as f:
        from ray import cloudpickle
        best_checkpoint = cloudpickle.load(f)

    # state_dict = trainer.latest_checkpoint["model_state_dict"]
    state_dict = best_checkpoint["model_state_dict"]
    model = TabNet(input_dim=len(features), output_dim=1)
    model.load_state_dict(state_dict)

    X_test_tensor = torch.Tensor(X_test.values)

    y_test = model(X_test_tensor).detach().numpy()
    tabnet_loss = mean_squared_error(y_true, y_test)


    print("Done!")
    print(results)

    print("\n========================")
    print(f"tabnet test loss: {tabnet_loss}")
    print(f"xgboost test loss: {xgb_loss}")
    print("========================")



