from typing import Callable, Optional
from ray.train.trainer import Trainer
from ray import train
from ray.train.session import get_session
from skorch import NeuralNet
from skorch.callbacks import Callback
from skorch.callbacks.logging import filter_log_keys
from torch.nn.parallel.distributed import DistributedDataParallel
import torch
from sklearn.base import clone
import io
from contextlib import AbstractContextManager


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


class TrainReportCallback(Callback):
    def __init__(
            self,
            keys_ignored=None,
    ):
        self.keys_ignored = keys_ignored

    def initialize(self):
        try:
            get_session()
        except ValueError:
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
        try:
            get_session()
        except ValueError:
            return
        history = net.history
        hist = history[-1]
        train.report(**{
            k: v
            for k, v in hist.items() if k in self._sorted_keys(hist.keys())
        })


class RayTrainNeuralNet(NeuralNet):
    @property
    def _default_callbacks(self):
        try:
            get_session()
            if train.world_rank() == 0:
                return super()._default_callbacks + [("ray_train",
                                                      TrainReportCallback())]
            return [("ray_train", TrainReportCallback())]
        except ValueError:
            return super()._default_callbacks + [("ray_train",
                                                  TrainReportCallback())]

    def initialize_module(self):
        super().initialize_module()
        try:
            get_session()
            self.module_ = DistributedDataParallel(
                self.module_,
                find_unused_parameters=True,
                device_ids=[train.local_rank()]
                if torch.cuda.is_available() else None)
        except ValueError:
            pass

    def initialize(self, initialize_ray=True):
        self.initialize_virtual_params()
        self.initialize_callbacks()

        if initialize_ray:
            try:
                get_session()
                self.initialize_criterion()
                self.initialize_module()
                self.initialize_optimizer()
                self.initialize_history()
            except ValueError:
                self.initialize_trainer()
        else:
            self.initialize_criterion()
            self.initialize_module()
            self.initialize_optimizer()
            self.initialize_history()

        self.initialized_ = True
        return self

    def initialize_trainer(self):
        self.trainer_ = Trainer("torch", num_workers=4, use_gpu=False)

    def _get_history_io(self, **values):
        return {
            "f_params": io.BytesIO(values.get("f_params", None)),
            "f_optimizer": io.BytesIO(values.get("f_optimizer", None)),
            "f_criterion": io.BytesIO(values.get("f_criterion", None)),
            "f_history": io.StringIO(values.get("f_history", None)),
        }

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        try:
            get_session()
            return super().fit_loop(X, y, epochs=epochs, **fit_params)
        except ValueError:

            est = clone(self)

            def train_func(config):
                est.fit(X, y, epochs=epochs, **fit_params)
                if train.world_rank() == 0:
                    output = self._get_history_io()
                    est.save_params(
                        f_params=output["f_params"],
                        f_optimizer=output["f_optimizer"],
                        f_criterion=output["f_criterion"],
                        f_history=output["f_history"])
                    return {k: v.getvalue() for k, v in output.items()}
                return {}

            with ray_trainer_start_shutdown(self.trainer_):
                results = self.trainer_.run(train_func)
            self.initialize(initialize_ray=False)
            params = results[0]
            self.module_ = params.pop("f_params")
            params = self._get_history_io(**params)
            params.pop("f_params")
            self.load_params(**params)

    def predict_proba(self, X):
        try:
            get_session()
            return super().predict_proba(X)
        except ValueError:
            est = clone(self)

            def train_func(config):
                config = self._get_history_io(**config)
                est.initialize(initialize_ray=False).load_params(**config)
                ret = est.predict_proba(X)
                if train.world_rank() == 0:
                    return {"ret": ret}
                return {}

            output = self._get_history_io()
            self.save_params(
                f_params=None,
                f_optimizer=output["f_optimizer"],
                f_criterion=output["f_criterion"],
                f_history=output["f_history"])
            output.pop("f_params")
            output = {k: v.getvalue() for k, v in output.items()}
            output["f_params"] = self.module_

            with ray_trainer_start_shutdown(self.trainer_):
                results = self.trainer_.run(train_func, output)
            return results[0]["ret"]
