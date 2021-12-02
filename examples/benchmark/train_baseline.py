import argparse
import os

import dask
import dask.dataframe as dd
import ray
import torch
from dask_ml.preprocessing import StandardScaler
from ray import train
from ray.train import Trainer
from ray.util.dask import ray_dask_get
from torch import nn
import time


from ray_sklearn.models.tabnet import TabNet

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
        "--num-epochs",
        type=int,
        default=5,
        help="Sets the number of training epochs. Defaults to 5.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="If enabled, train on a smaller set of data.",
    )
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
    smoke_test = args.smoke_test
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
    dask.config.set(scheduler=ray_dask_get)
    scaler = StandardScaler(copy=False)
    dask_df = scaler.fit_transform(
        dd.read_csv(os.path.expanduser("~/data/train.csv"))[
            features + [target]].dropna().astype("float32"))
    if smoke_test:
        dask_df = dask_df.sample(frac=0.1)
    dataset = ray.data.from_dask(dask_df)

    print(dataset)
    print("dataset loaded")

    dtypes = [torch.float] * len(features)

    split = 0.8
    split_index = int(dataset.count() * split)
    train_dataset, validation_dataset = dataset.random_shuffle().split_at_indices(
        [split_index])

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

        device = train.torch.get_device()

        results = []
        for i in range(num_epochs):
            train_dataset = next(train_dataset_iterator)
            validation_dataset = next(validation_dataset_iterator)
            train_torch_dataset = train_dataset.to_torch(
                label_column=target, batch_size=train_worker_batch_size)
            validation_torch_dataset = validation_dataset.to_torch(
                label_column=target, batch_size=train_worker_batch_size)

            last_time = time.time()

            model.train()
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

            last_time = time.time()

            model.eval()
            total_loss = 0
            num_batches = 0
            num_rows = 0
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
                    # total_loss += criterion(pred, y).item()
                    # num_batches += 1
                    total_loss += count * criterion(pred, y).item()
                    num_rows += count
            loss = total_loss / num_rows  # TODO: should this be num batches or num rows?
            result = {"mean square error": loss}
            print(f"Validation epoch: [{i}], mean square error:[{loss}]")
            results.append(result)

        return results

    train_start = time.time()

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    results = trainer.run(train_func, dataset=datasets)
    trainer.shutdown()


    train_end = time.time()
    train_time = train_end - train_start

    print(f"Training completed in {train_time} seconds.")


    print("Done!")
    print(results)
