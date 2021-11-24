import time
import os
import json
from queue import Queue

from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from ray import train

from skorch.callbacks import Callback, EpochTimer
from skorch.callbacks.logging import filter_log_keys

from ray_sklearn.skorch_approach.utils import (
    is_in_train_session, is_dataset_or_ray_dataset, is_using_gpu)
from ray_sklearn.skorch_approach.callbacks.constants import PROFILER_KEY


class RayTrainCallback(Callback):
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
        net.history.record('dur_s', time.time() - self.epoch_start_time_)


class PerformanceLogger(RayTrainCallback):
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
            'to_device_dur_s', self.X_to_device_time_ + self.y_to_device_time_)
        net.history.record_batch('forward_pass_dur_s', self.forward_pass_time_)
        net.history.record_batch('backward_pass_dur_s',
                                 self.backward_pass_time_)


class PytorchProfilerLogger(RayTrainCallback):
    def __init__(self, profiler_args=None, **kwargs) -> None:
        self.profiler_args = profiler_args
        super().__init__(**kwargs)

    def _trace_handler(self, p: profile):
        dir_name = "pytorch_profiler_trace"
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        filename = f"worker_{train.world_rank()}_{self.epoch_}.pt.trace.json"
        path = os.path.join(dir_name, filename)
        p.export_chrome_trace(path)
        with open(path) as f:
            data = f.read()
        self.profiler_traces_.append((filename, data, p.events()))

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.has_gpu_ = is_using_gpu(net.device)
        self.profiler_args_ = self.profiler_args or {
            "activities": [ProfilerActivity.CPU] + [ProfilerActivity.CUDA]
            if self.has_gpu_ else [],
            "with_stack": False,
            "schedule": schedule(wait=0, warmup=0, active=net.max_epochs),
            "on_trace_ready": self._trace_handler
        }
        self.epoch_ = 0
        self.record_functions_ = {}
        self.profiler_ = profile(**self.profiler_args_)
        self.profiler_.__enter__()

    def on_train_end(self, net, X=None, y=None, **kwargs):
        self.profiler_.__exit__(None, None, None)

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


class TrainReportCallback(RayTrainCallback):
    def __init__(
            self,
            keys_ignored=None,
    ):
        self.keys_ignored = keys_ignored

    def initialize(self):
        if not is_in_train_session():
            return
        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        #self.keys_ignored_.add("batches")
        return self

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur_s' is put last;
          * keys that start with 'event_' are put just before 'dur_s';
          * all remaining keys are sorted alphabetically.
        """
        sorted_keys = []

        # make sure "epoch" comes first
        if ("epoch" in keys) and ("epoch" not in self.keys_ignored_):
            sorted_keys.append("epoch")

        # ignore keys like *_best or event_*
        for key in filter_log_keys(
                sorted(keys), keys_ignored=self.keys_ignored_):
            if key != "dur_s":
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith("event_") and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure "dur" comes last
        if ("dur_s" in keys) and ("dur_s" not in self.keys_ignored_):
            sorted_keys.append("dur_s")

        return sorted_keys

    def on_epoch_end(self, net, **kwargs):
        if not is_in_train_session():
            return
        history = net.history
        hist = history[-1]
        train.report(**{
            k: v
            for k, v in hist.items() if k in self._sorted_keys(hist.keys())
        })
