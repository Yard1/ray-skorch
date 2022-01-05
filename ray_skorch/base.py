# Based on code from https://github.com/skorch-dev/skorch

# BSD 3-Clause License

# Copyright (c) 2017, Benjamin Bossan, Daniel Nouri, Marian Tietz
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import AbstractContextManager
import io
import inspect
import pandas as pd
import warnings

from ray import train
from ray.train.callbacks.callback import TrainingCallback
from ray.train.trainer import Trainer
import ray.data
import ray.data.impl.progress_bar
from ray.data.dataset import Dataset

from skorch import NeuralNet
from skorch.callbacks import PassthroughScoring
from skorch.callbacks.base import _issue_warning_if_on_batch_override
from skorch.dataset import Dataset as SkorchDataset, unpack_data
from skorch.history import History
from skorch.utils import to_tensor

import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from sklearn.base import clone

from ray_skorch.callbacks.train import (
    HistoryLoggingCallback, TableHistoryPrintCallback,
    DetailedHistoryPrintCallback, TBXProfilerCallback)
from ray_skorch.callbacks.skorch import (
    TrainSklearnCallback, TrainCheckpoint, TrainReportCallback,
    PerformanceLogger, EpochTimerS, PytorchProfilerLogger)
from ray_skorch.dataset import (FixedSplit, PipelineIterator,
                                RayPipelineDataset, dataset_factory)
from ray_skorch.docs import (set_ray_train_neural_net_docs,
                             set_worker_neural_net_docs)

from ray_skorch.utils import (add_callback_if_not_already_in,
                              is_in_train_session, is_dataset_or_ray_dataset,
                              get_params_io)

_warned = False


class ray_trainer_start_shutdown(AbstractContextManager):
    """Context manager to start and shutdown a ``Trainer``.

    Args:
        initialization_hook (Optional[Callable]): The function to call on
            each worker when it is instantiated.
    """

    def __init__(self,
                 trainer: Trainer,
                 initialization_hook: Optional[Callable] = None) -> None:
        self.trainer = trainer
        self.initialization_hook = initialization_hook

    def __enter__(self):
        """Starts a ``Trainer``."""
        self.trainer.start(self.initialization_hook)

    def __exit__(self, __exc_type, __exc_value, __traceback) -> None:
        """Shuts down the started ``Trainer``."""
        self.trainer.shutdown()


class _WorkerRayTrainNeuralNet(NeuralNet):
    # Docstring modified through set_worker_neural_net_docs

    def __init__(self,
                 module,
                 criterion,
                 optimizer=torch.optim.SGD,
                 lr=0.01,
                 max_epochs=10,
                 batch_size=128,
                 iterator_train=PipelineIterator,
                 iterator_valid=PipelineIterator,
                 dataset=dataset_factory,
                 train_split=None,
                 callbacks=None,
                 predict_nonlinearity="auto",
                 warm_start=False,
                 verbose=1,
                 device="cpu",
                 profile: bool = False,
                 save_checkpoints: bool = False,
                 ddp_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self.profile = profile
        self.save_checkpoints = save_checkpoints
        self.ddp_kwargs = ddp_kwargs
        super().__init__(
            module,
            criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            **kwargs)

    def on_forward_pass_begin(self, net, X=None, **kwargs):
        """Called at the beginning of forward pass."""

    def on_forward_pass_end(self, net, X=None, **kwargs):
        """Called at the end of forward pass."""

    def on_backward_pass_begin(self, net, X=None, y=None, **kwargs):
        """Called at the beginning of backward pass."""

    def on_backward_pass_end(self, net, X=None, y=None, **kwargs):
        """Called at the end of backward pass."""

    def on_X_to_device_begin(self, net, X=None, **kwargs):
        """Called at the beginning of host to device copy of X."""

    def on_X_to_device_end(self, net, X=None, **kwargs):
        """Called at the end of host to device copy of X."""

    def on_y_to_device_begin(self, net, y=None, **kwargs):
        """Called at the beginning of host to device copy of y."""

    def on_y_to_device_end(self, net, y=None, **kwargs):
        """Called at the end of host to device copy of y."""

    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in ``method_name`` with
        parameters specified in ``cb_kwargs``.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_epoch_begin
        * on_epoch_end
        * on_batch_begin
        * on_batch_end

        """
        # TODO: remove after some deprecation period, e.g. skorch 0.12
        if not self.history:  # perform check only at the start
            _issue_warning_if_on_batch_override(self.callbacks_)

        getattr(self, method_name)(self, **cb_kwargs)
        for _, cb in self.callbacks_:
            # Modified in ray-skorch
            func = getattr(cb, method_name, None)
            if func:
                getattr(cb, method_name)(self, **cb_kwargs)

    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimerS()),
            ("train_loss",
             PassthroughScoring(
                 name="train_loss",
                 on_train=True,
             )),
            ("valid_loss", PassthroughScoring(name="valid_loss", )),
        ]

    def initialize_callbacks(self):
        super().initialize_callbacks()
        if self.profile:
            performance_callback = PerformanceLogger()
            if add_callback_if_not_already_in("ray_performance_logger",
                                              performance_callback,
                                              self.callbacks_):
                performance_callback.initialize()
            profiler_callback = PytorchProfilerLogger()
            if add_callback_if_not_already_in("ray_pytorch_profiler_logger",
                                              profiler_callback,
                                              self.callbacks_):
                profiler_callback.initialize()
        checkpoint_callback = TrainCheckpoint(
            save_checkpoints=self.save_checkpoints)
        if add_callback_if_not_already_in(
                "ray_checkpoint", checkpoint_callback, self.callbacks_):
            checkpoint_callback.initialize()

        report_callback = TrainReportCallback()
        if add_callback_if_not_already_in("ray_report", report_callback,
                                          self.callbacks_):
            report_callback.initialize()

        # make sure the report callback is at the end
        if not isinstance(self.callbacks_[-1][-1], TrainReportCallback):
            report_callback = next(
                ((name, callback) for name, callback in self.callbacks_
                 if isinstance(callback, TrainReportCallback)), None)
            if report_callback is None:
                raise RuntimeError("TrainReportCallback missing")
            self.callbacks_.remove(report_callback)
            self.callbacks_.append(report_callback)
        return self

    # TODO make this a callback
    def wrap_module_in_ddp(self):
        if not isinstance(self.module_, DistributedDataParallel):
            ddp_kwargs = self.ddp_kwargs or {}
            ddp_kwargs = {**{"find_unused_parameters": True}, **ddp_kwargs}
            self.module_ = train.torch.prepare_model(
                self.module_, ddp_kwargs=ddp_kwargs)
        return self

    def initialize(self):
        assert is_in_train_session()  # TODO improve
        return super().initialize()

    @property
    def iterator_train_(self):
        return getattr(self, "_iterator_train_", None)

    @property
    def iterator_valid_(self):
        return getattr(self, "_iterator_valid_", None)

    @iterator_train_.setter
    def iterator_train_(self, val):
        self._iterator_train_ = val

    @iterator_valid_.setter
    def iterator_valid_(self, val):
        self._iterator_valid_ = val

    def get_iterator(self, dataset, training=False):
        if training:
            initalized_iterator = self.iterator_train_
            if initalized_iterator is None:
                kwargs = self.get_params_for("iterator_train")
                iterator = self.iterator_train
        else:
            initalized_iterator = self.iterator_valid_
            if initalized_iterator is None:
                kwargs = self.get_params_for("iterator_valid")
                iterator = self.iterator_valid

        if initalized_iterator is not None:
            return iter(initalized_iterator)

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = len(dataset)

        initalized_iterator = iterator(dataset, **kwargs)

        if training:
            self.iterator_train_ = initalized_iterator
        else:
            self.iterator_valid_ = initalized_iterator

        return initalized_iterator

    def fit(self, X, y=None, X_val=None, y_val=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(X, y, X_val=X_val, y_val=y_val, **fit_params)
        return self

    def partial_fit(self,
                    X,
                    y=None,
                    classes=None,
                    X_val=None,
                    y_val=None,
                    **fit_params):
        if not self.initialized_:
            self.initialize()

        self.notify("on_train_begin", X=X, y=y)
        self.wrap_module_in_ddp()
        try:
            self.fit_loop(X, y, X_val=X_val, y_val=y_val, **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify("on_train_end", X=X, y=y)
        return self

    def fit_loop(self,
                 X,
                 y=None,
                 epochs=None,
                 X_val=None,
                 y_val=None,
                 **fit_params):
        assert is_in_train_session()  # TODO improve
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        if X_val is None:
            dataset_train, dataset_valid = self.get_split_datasets(
                X, y, **fit_params)
        else:
            self.check_data(X_val, y_val)
            if is_dataset_or_ray_dataset(X_val) and y_val is None:
                y_val = y
            dataset_train = self.get_dataset(X, y)
            dataset_valid = self.get_dataset(X_val, y_val)

        assert dataset_train.y == dataset_valid.y  # TODO improve

        on_epoch_kwargs = {
            "dataset_train": dataset_train,
            "dataset_valid": dataset_valid,
        }

        for _ in range(epochs):
            self.notify("on_epoch_begin", **on_epoch_kwargs)

            self.run_single_epoch(
                dataset_train,
                training=True,
                prefix="train",
                step_fn=self.train_step,
                **fit_params)

            if dataset_valid is not None:
                self.run_single_epoch(
                    dataset_valid,
                    training=False,
                    prefix="valid",
                    step_fn=self.validation_step,
                    **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        self.notify("on_backward_pass_begin", X=Xi, y=yi)
        loss.backward()
        self.notify("on_backward_pass_end", X=Xi, y=yi)
        return {
            "loss": loss,
            "y_pred": y_pred,
        }

    def infer(self, x, **fit_params):
        self.notify("on_X_to_device_begin", X=x)
        x = to_tensor(x, device=self.device)
        self.notify("on_X_to_device_end", X=x)
        self.notify("on_forward_pass_begin", X=x)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            ret = self.module_(**x_dict)
        else:
            ret = self.module_(x, **fit_params)
        self.notify("on_forward_pass_end", X=x)
        if isinstance(ret, tuple):
            ret = ret[0]
        return ret

    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False):
        self.notify("on_y_to_device_begin", y=y_true)
        y_true = to_tensor(y_true, device=self.device)
        self.notify("on_y_to_device_end", y=y_true)
        return self.criterion_(y_pred, y_true)

    def predict_proba(self, X):
        self.wrap_module_in_ddp()
        return super().predict_proba(X)


set_worker_neural_net_docs(_WorkerRayTrainNeuralNet)


class RayTrainNeuralNet(NeuralNet):
    # Docstring modified through set_ray_train_neural_net_docs

    prefixes_ = NeuralNet.prefixes_ + ["worker_dataset"]

    def __init__(self,
                 module,
                 criterion,
                 num_workers: int,
                 optimizer=torch.optim.SGD,
                 lr=0.01,
                 max_epochs=10,
                 batch_size=128,
                 iterator_train=PipelineIterator,
                 iterator_valid=PipelineIterator,
                 dataset=dataset_factory,
                 worker_dataset=SkorchDataset,
                 train_split=FixedSplit(0.2),
                 callbacks: Union[List[Tuple[str, TrainSklearnCallback]], str,
                                  None] = None,
                 train_callbacks: Optional[List[Tuple[
                     str, TrainingCallback]]] = None,
                 predict_nonlinearity="auto",
                 warm_start=False,
                 verbose=1,
                 device="cpu",
                 trainer: Union[Type[Trainer], Trainer] = Trainer,
                 profile: bool = False,
                 save_checkpoints: bool = False,
                 ddp_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        global _warned
        if not _warned:
            _warned = True
            warnings.warn(
                "RayTrainNeuralNet and the rest of this package are "
                "experimental and not production ready. In particular, "
                "validation and error handling may be spotty. If you "
                "encounter any problems or have any suggestions, "
                "please open an issue on GitHub.")

        self.trainer = trainer
        self.worker_dataset = worker_dataset
        self.num_workers = num_workers
        self.profile = profile
        self.save_checkpoints = save_checkpoints
        self.train_callbacks = train_callbacks
        self.ddp_kwargs = ddp_kwargs
        super().__init__(
            module,
            criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            **kwargs)

    @property
    def latest_checkpoint_(self):
        """Same as ``self.trainer_.latest_checkpoint``."""
        self.check_is_fitted(["trainer_"])
        return self.trainer_.latest_checkpoint

    def initialize(self, initialize_ray=True):
        self._initialize_virtual_params()
        self._initialize_callbacks()

        if initialize_ray:
            self._initialize_trainer()
        else:
            self._initialize_module()
            self._initialize_criterion()
            self._initialize_optimizer()
            self._initialize_history()

            self._check_kwargs(self._kwargs)

        self.initialized_ = True
        return self

    def _initialize_callbacks(self):
        self.callbacks_ = []

    def _initialize_trainer(self):
        # this init context is for consistency and not being used at the moment
        with self._current_init_context("trainer"):
            if self.callbacks == "disable":
                self.callbacks_ = []
                return self
            self.initialize_trainer()
            return self

    def initialize_trainer(self):
        kwargs = self.get_params_for("trainer")
        trainer = self.trainer
        is_initialized = isinstance(trainer, Trainer)

        if kwargs or not is_initialized:

            kwargs["backend"] = "torch"
            if "num_workers" not in kwargs:
                kwargs["num_workers"] = self.num_workers
            if "use_gpu" not in kwargs:
                kwargs["use_gpu"] = self.device != "cpu"

            if is_initialized:
                trainer = type(trainer)

            if (is_initialized or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("trainer", kwargs)
                print(msg)

            trainer = trainer(**kwargs)

        if not isinstance(trainer._backend_config, train.torch.TorchConfig):
            raise ValueError("Only torch backend is supported")

        self.trainer_ = trainer
        return self

    def _create_worker_estimator(self) -> _WorkerRayTrainNeuralNet:
        """Create the worker estimator.

        This clones self, but removes all attributes that are not set
        in ``_WorkerRayTrainNeuralNet.__init__``, and then changes the
        base of the cloned object to ``_WorkerRayTrainNeuralNet``.
        """
        est = clone(self)
        worker_attributes = set(
            inspect.signature(_WorkerRayTrainNeuralNet.__init__).parameters)
        driver_attributes = set(
            inspect.signature(self.__class__.__init__).parameters)
        attributes_to_remove = driver_attributes.difference(worker_attributes)
        for attr in attributes_to_remove:
            del est.__dict__[attr]
        est.__class__ = _WorkerRayTrainNeuralNet
        est.set_params(dataset=self.worker_dataset, train_split=None)
        return est

    def get_default_train_callbacks(
            self) -> List[Tuple[str, TrainingCallback]]:
        callbacks = []
        if self.verbose <= 0:
            history_callback = HistoryLoggingCallback()
        else:
            history_callback = DetailedHistoryPrintCallback(
            ) if self.profile else TableHistoryPrintCallback()
        callbacks.append(("history_logger", history_callback))
        if self.profile:
            tbx_callback = TBXProfilerCallback()
            callbacks.append(("tbx_profiler_logger", tbx_callback))
        return callbacks

    def get_train_callbacks(self) -> List[TrainingCallback]:
        # TODO guard against duplicate keys
        # TODO do what initialize_callbacks does
        train_callbacks = self.train_callbacks or []
        # If we got a list of just callbacks, turn them into a (name, callback)
        # list
        self.train_callbacks_ = [
            callback
            if isinstance(callback, tuple) else (callback.__name__, callback)
            for callback in train_callbacks
        ]

        default_callbacks = self.get_default_train_callbacks()
        # put in default callbacks only if there are no
        # callbacks with their name or type already
        for name, callback in default_callbacks:
            add_callback_if_not_already_in(name, callback,
                                           self.train_callbacks_)
        return [callback for name, callback in self.train_callbacks_]

    def _create_train_function(self) -> Callable[[Dict[str, Any]], None]:
        """Create the Ray Train training function to be ran on all workers."""

        def train_func(config):
            label: str = config.pop("label")
            dataset_class: Type[RayPipelineDataset] = config.pop(
                "dataset_class")
            dataset_params: Dict[str, Any] = config.pop("dataset_params")
            estimator: _WorkerRayTrainNeuralNet = config.pop("estimator")
            epochs: int = config.pop("epochs")
            show_progress_bars: bool = config.pop("show_progress_bars")
            fit_params: Dict[str, Any] = config.pop("fit_params")
            ray.data.set_progress_bars(show_progress_bars)

            X_train = dataset_class(
                train.get_dataset_shard("dataset_train"), label)
            X_train.set_params(**dataset_params)
            try:
                X_val = dataset_class(
                    train.get_dataset_shard("dataset_valid"), label)
                X_val.set_params(**dataset_params)
            except KeyError:
                X_val = None

            original_device = estimator.device
            estimator.set_params(device=train.torch.get_device())
            estimator.fit(
                X_train, None, epochs=epochs, X_val=X_val, **fit_params)

            if train.world_rank() == 0:
                estimator.set_params(device=original_device)
                output = get_params_io()
                # get the module from inside DistributedDataParallel
                if isinstance(estimator.module_, DistributedDataParallel):
                    estimator.module_ = estimator.module_.module
                estimator.save_params(**output)
                output = {k: v.getvalue() for k, v in output.items()}
            else:
                output = {}
            output["history"] = estimator.history_
            return output

        return train_func

    def _create_prediction_function(self) -> Callable[[Dict[str, Any]], None]:
        """Create the Ray Train pred function to be ran on all workers."""

        def prediction_func(config):
            label: str = config.pop("label")
            dataset_class: Type[RayPipelineDataset] = config.pop(
                "dataset_class")
            dataset_params: Dict[str, Any] = config.pop("dataset_params")
            estimator: _WorkerRayTrainNeuralNet = config.pop("estimator")
            estimator_params: Dict[str, io.BytesIO] = get_params_io(
                **config.pop("estimator_params"))
            history: History = config.pop("history")
            show_progress_bars: bool = config.pop("show_progress_bars")
            ray.data.set_progress_bars(show_progress_bars)

            X = dataset_class(train.get_dataset_shard("dataset"), label)
            X.set_params(**dataset_params)

            estimator.set_params(device=train.torch.get_device())
            estimator.initialize().load_params(**estimator_params)
            estimator.history = history
            X_pred = estimator.predict_proba(X)
            X_pred = ray.data.from_pandas(pd.DataFrame(X_pred))
            return {"X_pred": X_pred}

        return prediction_func

    def fit(self,
            X,
            y=None,
            X_val=None,
            y_val=None,
            checkpoint=None,
            **fit_params):
        if self.warm_start:
            raise NotImplementedError(
                "`warm_start` parameter is not yet supported. If you want to "
                "resume training, pass a ray-skorch checkpoint as "
                "`checkpoint`.")

        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            checkpoint=checkpoint,
            **fit_params)
        return self

    def partial_fit(self,
                    X,
                    y=None,
                    classes=None,
                    X_val=None,
                    y_val=None,
                    checkpoint=None,
                    **fit_params):
        if not self.initialized_:
            self.initialize()

        self.notify("on_train_begin", X=X, y=y)
        try:
            self.fit_loop(
                X,
                y,
                X_val=X_val,
                y_val=y_val,
                checkpoint=checkpoint,
                **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify("on_train_end", X=X, y=y)
        return self

    def fit_loop(self,
                 X,
                 y=None,
                 epochs=None,
                 X_val=None,
                 y_val=None,
                 checkpoint=None,
                 **fit_params):
        dataset: Dict[str, Dataset] = {}

        if X_val is None:
            ray_dataset_train, ray_dataset_valid = self.get_split_datasets(
                X, y, **fit_params)
        else:
            self.check_data(X_val, y_val)
            if is_dataset_or_ray_dataset(X_val) and y_val is None:
                y_val = y
            ray_dataset_train = self.get_dataset(X, y)
            ray_dataset_valid = self.get_dataset(X_val, y_val)
            # TODO refactor
            if not isinstance(ray_dataset_train, RayPipelineDataset):
                params = ray_dataset_train.get_params()
                ray_dataset_train = RayPipelineDataset(
                    ray_dataset_train.X, y=ray_dataset_train.y)
                ray_dataset_train.set_params(**params)
            if not isinstance(ray_dataset_valid, RayPipelineDataset):
                params = ray_dataset_valid.get_params()
                ray_dataset_valid = RayPipelineDataset(
                    ray_dataset_valid.X, y=ray_dataset_valid.y)
                ray_dataset_valid.set_params(**params)

        dataset["dataset_train"] = ray_dataset_train.X

        if ray_dataset_valid is not None:
            assert ray_dataset_train.y == ray_dataset_valid.y  # TODO improve
            dataset["dataset_valid"] = ray_dataset_valid.X

        worker_estimator = self._create_worker_estimator()
        train_func = self._create_train_function()

        show_progress_bars = ray.data.impl.progress_bar._enabled
        callbacks = self.get_train_callbacks()

        with ray_trainer_start_shutdown(self.trainer_):
            results = self.trainer_.run(
                train_func,
                config={
                    "dataset_class": self.dataset,
                    "label": ray_dataset_train.y,
                    "dataset_params": ray_dataset_train.get_params(),
                    "estimator": worker_estimator,
                    "epochs": epochs,
                    "show_progress_bars": show_progress_bars,
                    "fit_params": fit_params
                },
                dataset=dataset,
                callbacks=callbacks,
                checkpoint=checkpoint)

        # get back params and history from rank 0 worker
        self.initialize(initialize_ray=False)
        params = results[0]
        self.history_ = params.pop("history")
        params = get_params_io(**params)
        self.load_params(**params)
        self.worker_histories_ = [self.history_] + [
            result["history"] for result in results[1:]
        ]

        history_logger = next(
            (callback for callback in callbacks
             if isinstance(callback, HistoryLoggingCallback)), None)
        if history_logger:
            self.ray_train_history_ = history_logger._history
        else:
            self.ray_train_history_ = None
        return self

    def predict_proba(self, X) -> Dataset:
        ray_dataset = RayPipelineDataset(
            X, y=None, random_shuffle_each_window=False)
        worker_estimator = self._create_worker_estimator()

        prediction_func = self._create_prediction_function()

        estimator_params = get_params_io()
        self.save_params(
            f_params=estimator_params["f_params"],
            f_optimizer=estimator_params["f_optimizer"],
            f_criterion=estimator_params["f_criterion"])
        estimator_params = {
            k: v.getvalue()
            for k, v in estimator_params.items()
        }

        show_progress_bars = ray.data.impl.progress_bar._enabled
        callbacks = self.get_train_callbacks()

        with ray_trainer_start_shutdown(self.trainer_):
            results = self.trainer_.run(
                prediction_func,
                config={
                    "dataset_class": self.dataset,
                    "label": ray_dataset.y,
                    "dataset_params": ray_dataset.get_params(),
                    "estimator": worker_estimator,
                    "estimator_params": estimator_params,
                    "history": self.history,
                    "show_progress_bars": show_progress_bars,
                },
                dataset={"dataset": ray_dataset.X},
                callbacks=callbacks,
            )
        datasets: List[ray.data.Dataset] = [
            result["X_pred"] for result in results
        ]
        first_dataset = datasets.pop(0)
        if datasets:
            return first_dataset.union(*datasets)
        return first_dataset


set_ray_train_neural_net_docs(RayTrainNeuralNet)
