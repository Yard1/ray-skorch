import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import ray
import torch
import xgboost_ray
from ray import train
from ray.train import Trainer, CheckpointStrategy, TrainingCallback
from ray.train.callbacks import TBXLoggerCallback
from sklearn.metrics import accuracy_score
from torch import nn
from xgboost_ray import RayDMatrix, RayParams

from ray_sklearn.models.tabnet import TabNet

ray.data.set_progress_bars(False)


num_workers = 1
use_gpu = False
address = None
num_epochs = 50
batch_size = 512
worker_batch_size = None
shuffle = False
lr = 0.02
debug = True

if worker_batch_size:
    print(f"Using worker batch size: {worker_batch_size}")
    train_worker_batch_size = worker_batch_size
else:
    train_worker_batch_size = batch_size / num_workers
    print(f"Using global batch size: {batch_size}. "
          f"For {num_workers} workers the per worker batch size is {train_worker_batch_size}.")

ray.init(address=address)

###############################################################################
# Prepare Dataset
###############################################################################

target = "hand"
data_path = os.path.expanduser("~/data/poker/train.csv")
dataset = ray.data.read_csv(data_path)

def preprocess(df):
    df = df - 1
    df[target] = df[target] + 1
    num_cols = pd.concat([df[col] for col in df.columns if col.startswith("C")],
                         axis=1)
    num_cols.columns = [f"n{col}" for col in num_cols.columns]
    df = pd.concat((df, num_cols), axis=1)
    return df

dataset = dataset.map_batches(preprocess, batch_format="pandas")
print(dataset)
"""
Dataset(num_blocks=1, num_rows=25010, schema={S1: int64, C1: int64, S2: int64, C2: int64, S3: int64, C3: int64, S4: int64, C4: int64, S5: int64, C5: int64, hand: int64, nC1: int64, nC2: int64, nC3: int64, nC4: int64, nC5: int64})
"""

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

test_df = test_dataset.to_pandas()
X_test = test_df.drop(target, axis=1)
y_true = test_df[target]

###############################################################################
# Prepare Model
###############################################################################

cat_params = {"cat_idxs": list(range(10)),
              "cat_dims": [4, 13] * 5,
              "cat_emb_dim": [4, 7] * 5
              }
tabnet_params = {"n_d": 16, "n_a": 8, "n_steps": 1,
                 "input_dim": 15, "output_dim": 10,
                 **cat_params}
print(f"tabnet_params: {tabnet_params}")
"""
tabnet_params: {
'n_d': 16, 
'n_a': 8, 
'n_steps': 1, 
'input_dim': 15, 
'output_dim': 10, 
'cat_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
'cat_dims': [4, 13, 4, 13, 4, 13, 4, 13, 4, 13], 
'cat_emb_dim': [4, 7, 4, 7, 4, 7, 4, 7, 4, 7]
}
"""

###############################################################################
# Distributed Training Function
###############################################################################

def train_func(config):
    print(datasets)
    train_dataset_pipeline = train.get_dataset_shard(
        "train_dataset_pipeline")
    validation_dataset_pipeline = train.get_dataset_shard(
        "validation_dataset_pipeline")
    train_dataset_iterator = train_dataset_pipeline.iter_epochs()
    validation_dataset_iterator = validation_dataset_pipeline.iter_epochs()

    model = TabNet(**tabnet_params)
    model = train.torch.prepare_model(model, ddp_kwargs={
        "find_unused_parameters": True})
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    device = train.torch.get_device()

    results = []
    for i in range(num_epochs):
        train_dataset = next(train_dataset_iterator)
        validation_dataset = next(validation_dataset_iterator)
        train_torch_dataset = train_dataset.to_torch(
            label_column=target,  batch_size=train_worker_batch_size)
        validation_torch_dataset = validation_dataset.to_torch(
            label_column=target,  batch_size=train_worker_batch_size)

        model.train()
        train_train_loss = 0
        train_num_rows = 0
        for batch_idx, (X, y) in enumerate(train_torch_dataset):

            X = X.to(device)
            y = y.to(device)
            y = y.squeeze(1)
            print(X[0])

            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = len(X)
            train_train_loss += count * loss.item()
            train_num_rows += count

        train_loss = train_train_loss / train_num_rows

        model.eval()
        val_total_loss = 0
        val_num_rows = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(validation_torch_dataset):
                X = X.to(device)
                y = y.to(device)
                y = y.squeeze(1)
                count = len(X)
                pred = model(X)
                val_total_loss += count * criterion(pred, y).item()
                val_num_rows += count
        val_loss = val_total_loss / val_num_rows

        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        train.report(train_loss=train_loss, val_loss=val_loss, lr=curr_lr)

        state_dict = model.state_dict()
        from torch.nn.modules.utils import \
            consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        train.save_checkpoint(val_loss=val_loss, model_state_dict=state_dict)
        results.append(val_loss)

    return results

###############################################################################
# Ray Train
###############################################################################

train_start = time.time()


class PrintingCallback(TrainingCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)


trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
trainer.start()
results = trainer.run(train_func, dataset=datasets,
                      callbacks=[TBXLoggerCallback(), PrintingCallback()],
                      checkpoint_strategy=CheckpointStrategy(
                          checkpoint_score_attribute="val_loss",
                          checkpoint_score_order="min"))
trainer.shutdown()

train_end = time.time()
train_time = train_end - train_start

print(f"Training completed in {train_time} seconds.")

bcp = trainer.best_checkpoint_path
print(f"Best Checkpoint Path: {bcp}")
with bcp.open("rb") as f:
    from ray import cloudpickle
    best_checkpoint = cloudpickle.load(f)

state_dict = best_checkpoint["model_state_dict"]
model = TabNet(**tabnet_params)
model.load_state_dict(state_dict)

X_test_tensor = torch.Tensor(X_test.values)

pred = model(X_test_tensor)
y_test = pred.detach().numpy()
y_test = np.argmax(y_test, axis=1)

print(f"y_true:{y_true}")
print(f"y_test:{y_test}")

tabnet_acc = accuracy_score(y_true, y_test)

print(results)



###############################################################################
# XGBoost-Ray
###############################################################################

train_set = RayDMatrix(train_dataset, target)

evals_result = {}
# Set XGBoost config.
xgboost_params = {
    "tree_method": "approx",
    "objective": "multi:softmax",
    "eval_metric": ["merror"],
    "num_class": 10
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

xgb_acc = accuracy_score(y_true, pred)
print(f"XGBoost test acc: {xgb_acc}")

###############################################################################
# Final Results
###############################################################################

print("\n================================")
print(f"TabNet test acc: {tabnet_acc}")
print(f"XGBoost test acc: {xgb_acc}")
print("\n================================")



