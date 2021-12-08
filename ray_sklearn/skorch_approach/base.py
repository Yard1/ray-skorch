from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import AbstractContextManager
import io
import numpy as np
import inspect

from ray import train
from ray.train.callbacks.callback import TrainingCallback
from ray.train.trainer import Trainer
from ray.train.session import get_session
import ray.data
import ray.data.impl.progress_bar
from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline

from skorch import NeuralNet
from skorch.callbacks import Callback
from skorch.callbacks import PassthroughScoring
from skorch.callbacks.base import _issue_warning_if_on_batch_override
from skorch.callbacks.logging import filter_log_keys
from skorch.dataset import Dataset as SkorchDataset, unpack_data
from skorch.history import History
from skorch.utils import is_dataset, to_tensor

import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from sklearn.base import clone

from ray_sklearn.skorch_approach.callbacks.train import (
    HistoryLoggingCallback, TableHistoryPrintCallback,
    DetailedHistoryPrintCallback, TBXProfilerCallback)
from ray_sklearn.skorch_approach.callbacks.skorch import (
    TrainSklearnCallback, TrainCheckpoint, TrainReportCallback,
    PerformanceLogger, EpochTimerS, PytorchProfilerLogger)
from ray_sklearn.skorch_approach.dataset import (FixedSplit, PipelineIterator,
                                                 dataset_factory)
from ray_sklearn.skorch_approach.utils import (
    add_callback_if_not_already_in, is_in_train_session,
    is_dataset_or_ray_dataset, is_using_gpu)


class ray_trainer_start_shutdown(AbstractContextManager):
    def __init__(self,
                 trainer: Trainer,
                 initialization_hook: Optional[Callable] = None) -> None:
        self.trainer = trainer
        self.initialization_hook = initialization_hook

    def __enter__(self):
        self.trainer.start(self.initialization_hook)

    def __exit__(self, __exc_type, __exc_value, __traceback) -> None:
        self.trainer.shutdown()


class _WorkerRayTrainNeuralNet(NeuralNet):
    """Internal use only. Estimator used inside each Train worker."""

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
                 train_split=FixedSplit(0.2),
                 callbacks=None,
                 predict_nonlinearity='auto',
                 warm_start=False,
                 verbose=1,
                 device='cpu',
                 profile: bool = False,
                 save_checkpoints: bool = False,
                 **kwargs):
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
        self.profile = profile
        self.save_checkpoints = save_checkpoints

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
            # Modified in train-sklearn
            func = getattr(cb, method_name, None)
            if func:
                getattr(cb, method_name)(self, **cb_kwargs)

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimerS()),
            ('train_loss',
             PassthroughScoring(
                 name='train_loss',
                 on_train=True,
             )),
            ('valid_loss', PassthroughScoring(name='valid_loss', )),
        ]

    def initialize_callbacks(self):
        super().initialize_callbacks()
        #if train.world_rank() != 0:
        #    self.callbacks_ = [
        #        callback_tuple for callback_tuple in self.callbacks_
        #        if getattr(callback_tuple[0], "_on_all_ranks", False)
        #    ]
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

    def initialize_module(self):
        super().initialize_module()
        self.module_ = train.torch.prepare_model(
            self.module_, ddp_kwargs=dict(find_unused_parameters=True))
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
                kwargs = self.get_params_for('iterator_train')
                iterator = self.iterator_train
        else:
            initalized_iterator = self.iterator_valid_
            if initalized_iterator is None:
                kwargs = self.get_params_for('iterator_valid')
                iterator = self.iterator_valid

        if initalized_iterator is not None:
            return iter(initalized_iterator)

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        if kwargs['batch_size'] == -1:
            kwargs['batch_size'] = len(dataset)

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

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, X_val=X_val, y_val=y_val, **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
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
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

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
        self.notify('on_backward_pass_begin', X=Xi, y=yi)
        loss.backward()
        self.notify('on_backward_pass_end', X=Xi, y=yi)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def infer(self, x, **fit_params):
        self.notify('on_X_to_device_begin', X=x)
        x = to_tensor(x, device=self.device)
        self.notify('on_X_to_device_end', X=x)
        self.notify('on_forward_pass_begin', X=x)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            ret = self.module_(**x_dict)
        else:
            ret = self.module_(x, **fit_params)
        self.notify('on_forward_pass_end', X=x)
        return ret

    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False):
        self.notify('on_y_to_device_begin', y=y_true)
        y_true = to_tensor(y_true, device=self.device)
        self.notify('on_y_to_device_end', y=y_true)
        return self.criterion_(y_pred, y_true)


class RayTrainNeuralNet(NeuralNet):
    prefixes_ = NeuralNet.prefixes_ + [
        "worker_dataset", "trainer", "train_callbacks"
    ]

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
                 predict_nonlinearity='auto',
                 warm_start=False,
                 verbose=1,
                 device='cpu',
                 trainer: Union[Type[Trainer], Trainer] = Trainer,
                 profile: bool = False,
                 save_checkpoints: bool = False,
                 **kwargs):
        self.trainer = trainer
        self.worker_dataset = worker_dataset
        self.num_workers = num_workers
        self.profile = profile
        self.save_checkpoints = save_checkpoints
        self.train_callbacks = train_callbacks
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
        with self._current_init_context('trainer'):
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

    def _get_params_io(self, only_keys=None, **values):
        ret = {
            "f_params": io.BytesIO(values.get("f_params", None)),
            "f_optimizer": io.BytesIO(values.get("f_optimizer", None)),
            "f_criterion": io.BytesIO(values.get("f_criterion", None)),
        }
        return {k: v for k, v in ret.items() if k in (only_keys or ret.keys())}

    def _get_worker_estimator(self) -> _WorkerRayTrainNeuralNet:
        est = clone(self)
        worker_attributes = set(
            inspect.signature(_WorkerRayTrainNeuralNet.__init__).parameters)
        driver_attributes = set(
            inspect.signature(self.__class__.__init__).parameters)
        attributes_to_remove = driver_attributes.difference(worker_attributes)
        for attr in attributes_to_remove:
            del est.__dict__[attr]
        est.__class__ = _WorkerRayTrainNeuralNet
        est.set_params(dataset=self.worker_dataset)
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

    def get_train_callbacks(self) -> List[Tuple[str, TrainingCallback]]:
        # TODO guard against duplicate keys
        # TODO do what initialize_callbacks does
        self.train_callbacks_ = [
            callback
            if isinstance(callback, tuple) else (callback.__name__, callback)
            for callback in self.train_callbacks
        ]
        default_callbacks = self.get_default_train_callbacks()
        for name, callback in default_callbacks:
            add_callback_if_not_already_in(name, callback,
                                           self.train_callbacks_)
        return self.train_callbacks_

    def fit(self,
            X,
            y=None,
            X_val=None,
            y_val=None,
            checkpoint=None,
            **fit_params):
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

        self.notify('on_train_begin', X=X, y=y)
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
        self.notify('on_train_end', X=X, y=y)
        return self

    def fit_loop(self,
                 X,
                 y=None,
                 epochs=None,
                 X_val=None,
                 y_val=None,
                 checkpoint=None,
                 **fit_params):
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

        est = self._get_worker_estimator()
        show_progress_bars = ray.data.impl.progress_bar._enabled

        def train_func(config):
            label = config.pop("label")
            dataset_class = config.pop("dataset_class")
            dataset_params = config.pop("dataset_params")
            ray.data.set_progress_bars(show_progress_bars)

            X_train = dataset_class(
                train.get_dataset_shard("dataset_train"), label)
            X_val = dataset_class(
                train.get_dataset_shard("dataset_valid"), label)
            X_train.set_params(**dataset_params)
            X_val.set_params(**dataset_params)

            original_device = est.device
            est.set_params(device=train.torch.get_device())
            est.fit(X_train, None, epochs=epochs, X_val=X_val, **fit_params)

            if train.world_rank() == 0:
                est.set_params(device=original_device)
                output = self._get_params_io()
                est.save_params(**output)
                output = {k: v.getvalue() for k, v in output.items()}
            else:
                output = {}
            output["history"] = est.history_
            return output

        callbacks = {k: v for k, v in self.get_train_callbacks()}
        with ray_trainer_start_shutdown(self.trainer_):
            results = self.trainer_.run(
                train_func,
                config={
                    "dataset_class": self.dataset,
                    "label": dataset_train.y,
                    "dataset_params": dataset_train.get_params(),
                },
                dataset={
                    "dataset_train": dataset_train.X,
                    "dataset_valid": dataset_valid.X
                },
                callbacks=list(callbacks.values()),
                checkpoint=checkpoint)

        # get back params and history from rank 0 worker
        self.initialize(initialize_ray=False)
        params = results[0]
        self.history_ = params.pop("history")
        self.module_ = params.pop("f_params")
        params = self._get_params_io(**params)
        params.pop("f_params")
        self.load_params(**params)
        self.worker_histories_ = [self.history_] + [
            result["history"] for result in results[1:]
        ]

        history_logger = next(
            (callback for callback in callbacks.values()
             if isinstance(callback, HistoryLoggingCallback)), None)
        if history_logger:
            self.ray_train_history_ = history_logger._history
        else:
            self.ray_train_history_ = None
        return self

    def predict_proba(self, X):
        raise NotImplementedError
        dataset = self.get_dataset(X, None)
        est = clone(self)

        def train_func(config):
            label = config.pop("label")
            config = self._get_history_io(**config)
            est.initialize(initialize_ray=False).load_params(**config)
            X_ray_dataset = train.get_dataset_shard().to_torch(
                label_column=label)
            ret = est.predict_proba(X_ray_dataset)
            return {"ret": ret}

        output = self._get_history_io()
        self.save_params(
            f_params=None,
            f_optimizer=output["f_optimizer"],
            f_criterion=output["f_criterion"],
            f_history=output["f_history"])
        output.pop("f_params")
        output = {k: v.getvalue() for k, v in output.items()}
        output["f_params"] = self.module_
        output["label"] = dataset.y

        with ray_trainer_start_shutdown(self.trainer_):
            results = self.trainer_.run(train_func, output, dataset=dataset.X)
        return np.vstack(
            [result["ret"].ravel().reshape(-1, 1) for result in results])