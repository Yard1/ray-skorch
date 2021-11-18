import time

from ray import train

from skorch.callbacks import Callback
from skorch.callbacks.logging import filter_log_keys

from ray_sklearn.skorch_approach.utils import (
    is_in_train_session, is_dataset_or_ray_dataset, is_using_gpu)


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
            'to_device_dur', self.X_to_device_time_ + self.y_to_device_time_)
        net.history.record_batch('forward_pass_dur', self.forward_pass_time_)
        net.history.record_batch('backward_pass_dur', self.backward_pass_time_)


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
        self.keys_ignored_.add("batches")
        return self

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur' is put last;
          * keys that start with 'event_' are put just before 'dur';
          * all remaining keys are sorted alphabetically.
        """
        sorted_keys = []

        # make sure "epoch" comes first
        if ("epoch" in keys) and ("epoch" not in self.keys_ignored_):
            sorted_keys.append("epoch")

        # ignore keys like *_best or event_*
        for key in filter_log_keys(
                sorted(keys), keys_ignored=self.keys_ignored_):
            if key != "dur":
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith("event_") and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure "dur" comes last
        if ("dur" in keys) and ("dur" not in self.keys_ignored_):
            sorted_keys.append("dur")

        return sorted_keys

    def on_epoch_end(self, net, **kwargs):
        if not is_in_train_session():
            return
        history = net.history
        hist = history[-1]
        train.report(history=hist, **{
            k: v
            for k, v in hist.items() if k in self._sorted_keys(hist.keys())
        })
