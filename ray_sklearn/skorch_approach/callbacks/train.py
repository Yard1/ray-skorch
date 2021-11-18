from typing import Callable, List, Dict, Iterable, Set, Union
import numpy as np
import numbers

from pprint import pprint
from ray.train.callbacks import TrainingCallback


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
    "valid_batch_size"
}


class PrintCallback(TrainingCallback):
    def __init__(self,
                 workers_to_log: Union[int, str, List[Union[
                     int, str]]] = "aggregate",
                 keys_to_ignore: Set[str] = DEFAULT_KEYS_TO_IGNORE,
                 aggregate_method: str = "nested",
                 aggregate_funcs: Dict[str, Callable[[List[
                     float]], float]] = DEFAULT_AGGREGATE_FUNC) -> None:
        self._workers_to_log = self._validate_workers_to_log(workers_to_log)
        self._keys_to_ignore = set(keys_to_ignore)
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

        for key, value in results[0].items():
            if key in self._keys_to_ignore:
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
            key: val
            for key, val in results_dict.items() if key in self._workers_to_log
        }
        pprint(print_dict)