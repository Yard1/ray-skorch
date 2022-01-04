import time
import os
import io
import pickle
from typing import (Any, Callable, Dict, Iterable, Optional, Union,
                    TYPE_CHECKING)
from skorch.callbacks.training import Checkpoint

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity

from ray import train

from skorch.callbacks import Callback, EpochTimer
from skorch.utils import _check_f_arguments, noop

from ray_skorch.utils import is_using_gpu
from ray_skorch.callbacks.constants import PROFILER_KEY
from ray_skorch.callbacks.utils import SortedKeysMixin

if TYPE_CHECKING:
    from ray_skorch.base import _WorkerRayTrainNeuralNet  # noqa: F401


class TrainSklearnCallback(Callback):
    """Abstract extension of the skorch ``Callback`` class.

    Adds several new notify points."""

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


class EpochTimerS(EpochTimer):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur_s``.

    """

    def on_epoch_end(self, net, **kwargs):
        net.history.record("dur_s", time.time() - self.epoch_start_time_)


class PerformanceLogger(TrainSklearnCallback):
    """Logs times taken for several operations.

    Operations logged:
        * Batch
        * Forward pass
        * Backward pass
        * X to device
        * y to device
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def on_forward_pass_begin(self, net, X=None, **kwargs):
        self.forward_pass_time_ = time.time()

    def on_forward_pass_end(self, net, X=None, **kwargs):
        self.forward_pass_time_ = time.time() - self.forward_pass_time_

    def on_backward_pass_begin(self, net, X=None, y=None, **kwargs):
        self.backward_pass_time_ = time.time()

    def on_backward_pass_end(self, net, X=None, y=None, **kwargs):
        self.backward_pass_time_ = time.time() - self.backward_pass_time_

    def on_X_to_device_begin(self, net, X=None, **kwargs):
        self.X_to_device_time_ = time.time()

    def on_X_to_device_end(self, net, X=None, **kwargs):
        self.X_to_device_time_ = time.time() - self.X_to_device_time_

    def on_y_to_device_begin(self, net, y=None, **kwargs):
        self.y_to_device_time_ = time.time()

    def on_y_to_device_end(self, net, y=None, **kwargs):
        self.y_to_device_time_ = time.time() - self.y_to_device_time_

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        net.history.record_batch(
            "to_device_dur_s", self.X_to_device_time_ + self.y_to_device_time_)
        net.history.record_batch("forward_pass_dur_s", self.forward_pass_time_)
        net.history.record_batch("backward_pass_dur_s",
                                 self.backward_pass_time_)


class PytorchProfilerLogger(TrainSklearnCallback):
    """Saves the profiler state to worker history so that it can be retrieved
    by Ray Train during ``train.report()`` (through ``TrainReportCallback``).

    Saves and reports the trace at training end.

    Operations logged:
        * Batch
        * Forward pass
        * Backward pass
        * X to device
        * y to device

    Args:
        profiler_args (Optional[Dict[str, Any]]): kwargs passed to
        ``torch.profiler.profile()`
    """

    def __init__(self,
                 profiler_args: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
        self.profiler_args = profiler_args
        super().__init__(**kwargs)

    def _trace_handler(self, p: profile):
        dir_name = "pytorch_profiler_trace"
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        filename = f"worker_{self.worker_rank_}_{self.epoch_}.pt.trace.json"
        path = os.path.join(dir_name, filename)
        # TODO consider compression
        try:
            p.export_chrome_trace(path)
            with open(path) as f:
                data = f.read()
            self.profiler_traces_.append((filename, data, p.events()))
        except RuntimeError:
            # trace is already saved
            pass

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.has_gpu_ = is_using_gpu(net.device)
        profiler_args = self.profiler_args or {}
        self.profiler_args_ = {**{
            "activities": [ProfilerActivity.CPU] + [ProfilerActivity.CUDA]
            if self.has_gpu_ else [],
            "with_stack": False,
            # "schedule": schedule(wait=0, warmup=1, active=4),
            "on_trace_ready": self._trace_handler
        }, **profiler_args}
        self.worker_rank_ = train.world_rank()
        self.epoch_ = 0
        self.record_functions_ = {}
        self.profiler_ = profile(**self.profiler_args_)
        self.profiler_.__enter__()
        self.profiler_is_initialized_ = True

    def on_train_end(self, net, X=None, y=None, **kwargs):
        if self.profiler_is_initialized_:
            self.profiler_.__exit__(None, None, None)
            self._trace_handler(self.profiler_)
            self.profiler_is_initialized_ = False

    def on_forward_pass_begin(self, net, X=None, **kwargs):
        record_name = "forward_pass"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_forward_pass_end(self, net, X=None, **kwargs):
        record_name = "forward_pass"
        self.record_functions_[record_name].__exit__(None, None, None)

    def on_backward_pass_begin(self, net, X=None, y=None, **kwargs):
        record_name = "backward_pass"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_backward_pass_end(self, net, X=None, y=None, **kwargs):
        record_name = "backward_pass"
        self.record_functions_[record_name].__exit__(None, None, None)

    def on_X_to_device_begin(self, net, X=None, **kwargs):
        record_name = "X_to_device"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_X_to_device_end(self, net, X=None, **kwargs):
        record_name = "X_to_device"
        self.record_functions_[record_name].__exit__(None, None, None)

    def on_y_to_device_begin(self, net, y=None, **kwargs):
        record_name = "y_to_device"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_y_to_device_end(self, net, y=None, **kwargs):
        record_name = "y_to_device"
        self.record_functions_[record_name].__exit__(None, None, None)

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        record_name = "batch"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        record_name = "batch"
        self.record_functions_[record_name].__exit__(None, None, None)

    def on_epoch_begin(self,
                       net,
                       dataset_train=None,
                       dataset_valid=None,
                       **kwargs):
        self.profiler_traces_ = []
        record_name = "epoch"
        self.record_functions_[record_name] = record_function(
            record_name).__enter__()

    def on_epoch_end(self,
                     net,
                     dataset_train=None,
                     dataset_valid=None,
                     **kwargs):
        self.epoch_ += 1
        record_name = "epoch"
        self.record_functions_[record_name].__exit__(None, None, None)
        self.profiler_.step()
        net.history.record(
            PROFILER_KEY, self.profiler_traces_
            if self.profiler_traces_ else [])


def default_monitor(net):
    return True


class TrainCheckpoint(Checkpoint, TrainSklearnCallback):
    """Save and load Ray Train checkpoints.

    By default, the checkpoint is saved every epoch. The behavior can
    be modified by setting the ``monitor`` argument.

    Args:
        monitor (Union[str, Callable[["_WorkerRayTrainNeuralNet"], bool]]):
            If callable, an epoch will be saved whenever it returns True.
            If string, an epoch will be saved if the metric with that name
            improves.
        f_params (bool): Whether to save the model parameters in the
            checkpoint. Defaults to True.
        f_optimizer (bool): Whether to save the optimizer state in the
            checkpoint. Defaults to True.
        f_criterion (bool): Whether to save the criterion state in the
            checkpoint. Defaults to True.
        f_history (bool): Whether to save the head worker history in the
            checkpoint. Defaults to True.
        f_pickle (bool): Whether to save the entire ``NeuralNet`` as a
            pickle in the checkpoint. This is usually not necessary,
            as all the necessary information is saved with other items,
            but may be useful for debugging. Defaults to False.
        event_name (str):
            Name of event to be placed in history when checkpoint is
            triggered. Pass ``None`` to disable placing events in history.
        save_checkpoints (bool):
            Whether to save checkpoints at all. Defaults to True.
        load_checkpoint (bool):
            Whether to load a checkpoint if provided. Defaults to True.
        sink (callable):
            The target that the information about created checkpoints is
            sent to. This can be a logger or ``print`` function (to send to
            stdout). By default the output is discarded.
        **kwargs:
            If you've created a custom module, e.g. ``net.mymodule_``, you
            can save that as well by passing ``f_mymodule=True``.
    """

    def __init__(self,
                 monitor: Union[str, Callable[["_WorkerRayTrainNeuralNet"],
                                              bool]] = default_monitor,
                 f_params: bool = True,
                 f_optimizer: bool = True,
                 f_criterion: bool = True,
                 f_history: bool = True,
                 f_pickle: bool = False,
                 event_name: str = "event_cp",
                 save_checkpoints: bool = True,
                 load_checkpoint: bool = True,
                 sink: Callable = noop,
                 **kwargs):
        self.monitor = monitor
        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.f_criterion = f_criterion
        self.f_history = f_history
        self.f_pickle = f_pickle
        self.event_name = event_name
        self.sink = sink
        self.load_checkpoint = load_checkpoint
        self.save_checkpoints = save_checkpoints
        self._check_kwargs(kwargs)
        vars(self).update(**kwargs)

    def initialize(self):
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        if not self.load_checkpoint:
            return
        checkpoint = train.load_checkpoint()
        if not checkpoint:
            return
        self._sink(f"Checkpoint found, loading...", net.verbose)

        try:
            epoch = checkpoint.pop("epoch")
            keys = checkpoint.pop("_keys")
        except KeyError:
            raise ValueError(
                "Invalid checkpoint. Ensure the checkpoint was created with "
                "ray-skorch. Expected 'epoch' and '_keys' keys, got "
                f"{list(checkpoint.keys())}.")
        checkpoint_params = {
            k: self._get_io(k, v)
            for k, v in checkpoint.items() if k in keys
        }
        net.load_params(**checkpoint_params)
        self._sink(
            f"Loaded checkpoint {list(checkpoint_params.keys())} "
            f"with epoch {epoch}", net.verbose)
        return

    def on_train_end(self, net, **kwargs):
        return

    def on_epoch_end(self, net, **kwargs):
        if not self.save_checkpoints:
            return
        return super().on_epoch_end(net, **kwargs)

    def save_model(self, net):
        """Save the model.

        This function saves some or all of the following:

          - model parameters;
          - optimizer state;
          - criterion state;
          - training history;
          - custom modules;
          - entire model object.

        """
        kwargs_module, kwargs_other = _check_f_arguments(
            self.__class__.__name__, **self._f_kwargs())

        params = {}

        # ensure a non DDP-wrapped module is saved
        ddp_module = None
        if isinstance(net.module_, DistributedDataParallel):
            ddp_module = net.module_
            net.module_ = net.module_.module

        for key, val in kwargs_module.items():
            if val is None:
                continue

            f = self._get_io(f"f_{key}")
            key = key[:-1]  # remove trailing "_"
            params[f"f_{key}"] = self._save_params(f, net, f"f_{key}",
                                                   f"{key} state")

        f_history = kwargs_other.get("f_history")
        if f_history:
            f = self.f_history_
            params["f_history"] = self._save_params(f, net, "f_history",
                                                    "history")

        f_pickle = kwargs_other.get("f_pickle")
        if f_pickle:
            f_pickle = self._get_io("f_pickle")
            with open(f_pickle, "wb") as f:
                pickle.dump(net, f)
            params["f_pickle"] = f_pickle

        if ddp_module:
            net.module_ = ddp_module

        epoch = net.history[-1]["epoch"]
        keys_to_load = tuple(key for key in params.keys() if key != "f_pickle")
        train.save_checkpoint(
            _keys=keys_to_load,
            epoch=epoch,
            **{k: v.getvalue()
               for k, v in params.items() if v})

    def _save_params(self, f, net, f_name, log_name):
        try:
            net.save_params(**{f_name: f})
            return f
        except Exception as e:  # pylint: disable=broad-except
            self._sink(
                "Unable to save {} to {}, {}: {}".format(
                    log_name, f,
                    type(e).__name__, e), net.verbose)

    def _validate_filenames(self):
        return

    @property
    def f_history_(self):
        # This is a property and not in initialize to allow ``NeuralNet``
        # to call ``load_params`` without needing the checkpoint to
        # by initialized.
        if self.f_history is None:
            return None
        return self._get_io("f_history")

    def _get_io(self, f_key, value=None):
        if f_key == "f_history":
            return io.StringIO(value)
        return io.BytesIO(value)


class TrainReportCallback(SortedKeysMixin, TrainSklearnCallback):
    """Report the last history entry from a worker to Ray Train.

    Args:
        keys_ignored (Optional[Union[str, Iterable]]):
            Keys in a history entry to ignore during reporting.
    """

    def __init__(
            self,
            keys_ignored: Optional[Union[str, Iterable]] = None,
    ):
        self.keys_ignored = keys_ignored

    def initialize(self):
        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        return self

    def on_epoch_end(self, net, **kwargs):
        history = net.history[-1]
        train.report(
            **{
                k: v
                for k, v in history.items() if k in self._sorted_keys(
                    history.keys(), self.keys_ignored_, filter_keys=False)
            })
