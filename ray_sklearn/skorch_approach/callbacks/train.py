from pathlib import Path
from typing import Callable, List, Dict, Iterable, Set, Union, Optional
import numpy as np
import numbers

from torch.profiler import profile, record_function, ProfilerActivity, schedule

from pprint import pprint
from ray.train.callbacks import TrainingCallback
from ray.train.callbacks.logging import TrainingSingleFileLoggingCallback
from ray_sklearn.skorch_approach.callbacks.constants import PROFILER_KEY


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

DEFAULT_KEYS_TO_NOT_AGGREGATE = {
    "epoch", "_timestamp", "_training_iteration", "train_batch_size",
    "valid_batch_size", PROFILER_KEY
}

DEFAULT_KEYS_TO_NOT_PRINT = {PROFILER_KEY}


class TBXProfilerCallback(TrainingSingleFileLoggingCallback):
    _default_filename = None

    def __init__(
            self,
            profiler_key: str = PROFILER_KEY,
            logdir: Optional[str] = None,
            filename: Optional[str] = None,
            workers_to_log: Optional[Union[int, List[int]]] = None) -> None:
        self.profiler_key = profiler_key
        super().__init__(
            logdir=logdir, filename=filename, workers_to_log=workers_to_log)

    def _create_log_path(self, logdir_path: Path, filename: Path) -> Path:
        return logdir_path

    def handle_result(self, results: List[Dict], **info):
        if not results:
            return
        results_to_log = {
            i: result
            for i, result in enumerate(results)
            if  not self._workers_to_log or i in self._workers_to_log
        }
        for i, result in results_to_log.items():
            if PROFILER_KEY in result and result[PROFILER_KEY]:
                for trace in result[PROFILER_KEY]:
                    name, data, _ = trace
                    with open(self.logdir.joinpath(Path(name)), "w") as f:
                        f.write(data)


class PrintCallback(TrainingCallback):
    def __init__(
            self,
            workers_to_log: Union[int, str, List[Union[int,
                                                       str]]] = "aggregate",
            keys_to_not_print: Set[str] = DEFAULT_KEYS_TO_NOT_AGGREGATE,
            keys_to_not_aggregate: Set[str] = DEFAULT_KEYS_TO_NOT_AGGREGATE,
            aggregate_method: str = "nested",
            aggregate_funcs: Dict[str, Callable[[List[
                float]], float]] = DEFAULT_AGGREGATE_FUNC) -> None:
        self._workers_to_log = self._validate_workers_to_log(workers_to_log)
        self._keys_to_not_print = set(keys_to_not_print)
        self._keys_to_not_aggregate = set(keys_to_not_aggregate)
        assert aggregate_method in ("nested", "flat")
        self._aggregate_method = aggregate_method
        self._aggregate_funcs = aggregate_funcs
        self._log_path = None
        self._history = []

    def _validate_workers_to_log(self, workers_to_log) -> List[int]:
        if not isinstance(workers_to_log, list):
            workers_to_log = [workers_to_log]

        if not isinstance(workers_to_log, Iterable):
            raise TypeError("workers_to_log must be an Iterable, got "
                            f"{type(workers_to_log)}.")
        if not all(
                isinstance(worker, int) or worker == "aggregate"
                for worker in workers_to_log):
            raise TypeError(
                "All elements of workers_to_log must be integers or 'aggregate'."
            )
        if len(workers_to_log) < 1:
            raise ValueError(
                "At least one worker must be specified in workers_to_log.")
        return workers_to_log

    def _get_aggregate_results(self, results: List[Dict]) -> Dict:
        aggregate_results = {}

        def _set_aggregate_key(aggregate_results: Dict, key: str,
                               aggregate_key: List[float]):
            if self._aggregate_method == "nested":
                aggregate = {}
                for func_key, func in self._aggregate_funcs.items():
                    aggregate[func_key] = func(aggregate_key)
                aggregate_results[key] = aggregate
            elif self._aggregate_method == "flat":
                for func_key, func in self._aggregate_funcs.items():
                    aggregate_results[f"{key}_{func_key}"] = func(
                        aggregate_key)

        if not isinstance(results[0], dict):
            return aggregate_results

        for key, value in results[0].items():
            if key in self._keys_to_not_aggregate:
                aggregate_results[key] = value
            elif isinstance(value, list):
                aggregate_results[key] = []
                for i, entry in enumerate(value):
                    aggregate_results[key].append(
                        self._get_aggregate_results([
                            result[key][i] for result in results
                            if key in result
                        ]))
            elif isinstance(value, numbers.Number):
                aggregate_key = [
                    result[key] for result in results if key in result
                ]
                _set_aggregate_key(aggregate_results, key, aggregate_key)
        return aggregate_results

    def handle_result(self, results: List[Dict], **info):
        results_dict = {idx: val for idx, val in enumerate(results)}
        if "aggregate" in self._workers_to_log:
            results_dict["aggregate"] = self._get_aggregate_results(results)
        self._history.append(results_dict)
        print_dict = {
            worker_rank: {
                k: v
                for k, v in worker_results.items()
                if k not in self._keys_to_not_print
            }
            for worker_rank, worker_results in results_dict.items()
            if worker_rank in self._workers_to_log
        }
        pprint(print_dict)
