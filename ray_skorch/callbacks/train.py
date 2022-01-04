from pathlib import Path
from typing import Any, Callable, List, Dict, Iterable, Set, Union, Optional
from itertools import cycle
from tabulate import tabulate
import numpy as np
import numbers
import sys

from numbers import Number
from skorch.utils import Ansi
from pprint import pprint
from ray.train.callbacks import TrainingCallback
from ray.train.callbacks.logging import TrainingSingleFileLoggingCallback
from ray_skorch.callbacks.constants import PROFILER_KEY, AGGREGATE_KEY
from ray_skorch.callbacks.utils import SortedKeysMixin


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
    "valid_batch_size", PROFILER_KEY, "valid_loss_best", "train_loss_best",
    "train_batch_count", "valid_batch_count"
}

DEFAULT_KEYS_TO_NOT_PRINT = {PROFILER_KEY, "batches"}

DEFAULT_KEYS_TO_NOT_PRINT_TABLE = DEFAULT_KEYS_TO_NOT_PRINT.union(
    {"_timestamp", "_time_this_iter_s", "_training_iteration"})


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
            if not self._workers_to_log or i in self._workers_to_log
        }
        for i, result in results_to_log.items():
            if PROFILER_KEY in result and result[PROFILER_KEY]:
                for trace in result[PROFILER_KEY]:
                    name, data, _ = trace
                    with open(self.logdir.joinpath(Path(name)), "w") as f:
                        f.write(data)


class HistoryLoggingCallback(TrainingCallback):
    def __init__(
            self,
            workers_to_log: Union[int, str, List[Union[int,
                                                       str]]] = AGGREGATE_KEY,
            *,
            keys_to_not_aggregate: Set[str] = DEFAULT_KEYS_TO_NOT_AGGREGATE,
            aggregate_method: str = "nested",
            aggregate_funcs: Dict[str, Callable[[List[
                float]], float]] = DEFAULT_AGGREGATE_FUNC) -> None:
        self._workers_to_log = self._validate_workers_to_log(workers_to_log)
        self._keys_to_not_aggregate = set(keys_to_not_aggregate or {})
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
                isinstance(worker, int) or worker == AGGREGATE_KEY
                for worker in workers_to_log):
            raise TypeError("All elements of workers_to_log must be integers "
                            "or 'aggregate'.")
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
        if AGGREGATE_KEY in self._workers_to_log:
            results_dict[AGGREGATE_KEY] = self._get_aggregate_results(results)
        self._history.append(results_dict)


class AbstractPrintCallback(HistoryLoggingCallback):
    def __init__(
            self,
            workers_to_log: Union[int, str, List[Union[int,
                                                       str]]] = AGGREGATE_KEY,
            *,
            keys_to_not_print: Set[str] = DEFAULT_KEYS_TO_NOT_PRINT,
            keys_to_not_aggregate: Set[str] = DEFAULT_KEYS_TO_NOT_AGGREGATE,
            aggregate_method: str = "nested",
            aggregate_funcs: Dict[str, Callable[[List[
                float]], float]] = DEFAULT_AGGREGATE_FUNC,
            sink: Callable[[Any], None] = pprint) -> None:
        self.keys_to_not_print = set(keys_to_not_print or {})
        self.sink = sink
        super().__init__(
            workers_to_log=workers_to_log,
            keys_to_not_aggregate=keys_to_not_aggregate,
            aggregate_method=aggregate_method,
            aggregate_funcs=aggregate_funcs)

    def handle_result(self, results: List[Dict], **info):
        super().handle_result(results, **info)
        self.display()

    def display(self):
        raise NotImplementedError

    def _sink(self, text, verbose=True):
        if (self.sink is not print) or verbose:
            self.sink(text)


class DetailedHistoryPrintCallback(SortedKeysMixin, AbstractPrintCallback):
    def display(self):
        print_dict = {
            worker_rank: {
                k: v
                for k, v in worker_results.items() if k in self._sorted_keys(
                    worker_results.keys(), self.keys_to_not_print)
            }
            for worker_rank, worker_results in self._history[-1].items()
            if worker_rank in self._workers_to_log
        }
        self._sink(print_dict)


# TODO _best keys should be calculated from aggregate
# and not taken from rank 0
class TableHistoryPrintCallback(SortedKeysMixin, AbstractPrintCallback):
    def __init__(
            self,
            workers_to_log: Union[int, str, List[Union[int,
                                                       str]]] = AGGREGATE_KEY,
            *,
            keys_to_not_print: Set[str] = DEFAULT_KEYS_TO_NOT_PRINT_TABLE,
            keys_to_not_aggregate: Set[str] = DEFAULT_KEYS_TO_NOT_AGGREGATE,
            aggregate_method: str = "nested",
            aggregate_funcs: Dict[str, Callable[[List[
                float]], float]] = DEFAULT_AGGREGATE_FUNC,
            aggregate_key_to_print: str = "mean",
            sink: Callable[[Any], None] = print,
            tablefmt="simple",
            floatfmt=".4f",
            stralign="right",
    ) -> None:
        self.aggregate_key_to_print = aggregate_key_to_print
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign
        super().__init__(
            workers_to_log=workers_to_log,
            keys_to_not_aggregate=keys_to_not_aggregate,
            aggregate_method=aggregate_method,
            aggregate_funcs=aggregate_funcs,
            sink=sink,
            keys_to_not_print=keys_to_not_print)
        self.first_iteration_ = True

    def display(self):
        if AGGREGATE_KEY in self._history[-1]:
            rank = AGGREGATE_KEY
        else:
            rank = 0

        data = {
            k: v
            for k, v in self._history[-1][rank].items()
            if k not in self.keys_to_not_print
        }

        if rank == AGGREGATE_KEY:
            data = self.handle_aggregate_results(data)

        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split("\n", 2)[:2]
            self._sink(header)
            self._sink(lines)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit("\n", 1)[-1])
        if self.sink is print:
            sys.stdout.flush()

    def handle_aggregate_results(self, aggregate_results: dict) -> dict:
        if self._aggregate_method == "nested":
            return {
                k: (v[self.aggregate_key_to_print]
                    if isinstance(v, dict) else v)
                for k, v in aggregate_results.items()
            }
        return aggregate_results

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable).

        """
        value = row[key]

        if isinstance(value, bool) or value is None:
            return "+" if value else ""

        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = "{}" if is_integer else "{:" + self.floatfmt + "}"

        # if numeric, there could be a "best" key
        key_best = key + "_best"
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        for key, color in zip(
                self._sorted_keys(row.keys(), self.keys_to_not_print), colors):
            formatted = self.format_row(row, key, color=color)
            if key.startswith("event_"):
                key = key[6:]
            yield key, formatted

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )
