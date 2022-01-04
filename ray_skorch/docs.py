from skorch.net import NeuralNet

from ray_skorch.utils import insert_before_substring

_docstring_neural_net_ray_args = """    num_workers : int
      Number of Ray Train workers to use.

"""

_docstring_neural_net_ray_worker_dataset = """    worker_dataset : torch Dataset (default=skorch.dataset.Dataset)
      Same as ``dataset``, but used internally inside workers. Should use torch Tensors.

"""  # noqa: E501

_docstring_neural_net_ray_train_callbacks = """    train_callbacks : None or list of tuples (str, Ray TrainingCallback instance) (default=None)
      List of Ray Train callbacks to enable in addition to necessary default callbacks. If a
      tuple with the same name or the same callback type as one of the default callbacks is passed,
      the default callback will be overriden.

"""  # noqa: E501

_docstring_neural_net_ray_trainer = """    trainer : ray.train.Trainer (class or instance) (default=ray.train.Trainer)
      The Ray Train Trainer to use. If a class is passed, it will be instantiated internally.

"""  # noqa: E501

_docstring_neural_net_ray_trainer = """    trainer : ray.train.Trainer (class or instance) (default=ray.train.Trainer)
      The Ray Train Trainer to use. If a class is passed, it will be instantiated internally.

"""  # noqa: E501

_docstring_neural_net_ray_ray_train_history = """    ray_train_history_ : list of dicts
      Histories from all Ray Train workers. A list of dicts. Each dict represents
      another epoch in order, with keys being the rank of the worker. Will be None
      if no Ray Train callback was an instance of ``HistoryLoggingCallback``.

"""  # noqa: E501

_docstring_neural_net_ray_kwargs = """    profile : bool (default=False)
      Whether to enable PyTorch Profiler and callbacks necessary
      for reporting.

    save_checkpoints : bool (default=True)
      Whether to enable or disable saving checkpoints (through
      a callback).

"""  # noqa: E501

_docstring_neural_net_ray_fit_params = """        X_val : validation data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If not
          provided, validation set will be obtained through the
          ``train_split``.

        y_val : validation target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If not
          provided, validation set will be obtained through the
          ``train_split``.

        checkpoint : dict or None (default=None)
          ray-skorch checkpoint to resume the training from (if
          ``load_checkpoint`` parameter is set to True).

"""  # noqa: E501


def set_ray_train_neural_net_docs(ray_train_neural_net_class):
    ray_train_neural_net_class.__doc__ = NeuralNet.__doc__.replace(
        "NeuralNet base class.",
        ("Distributed Skorch NeuralNet with Ray Train. "
         "Experimental and not production ready.")
    ).replace(
        ("train_split : None or callable (default=skorch.dataset."
         "ValidSplit(5))"), ("train_split : None or callable "
                             "(default=ray_skorch.dataset.FixedSplit(0.2))")
    ).replace(
        """    By default an :class:`.EpochTimer`, :class:`.BatchScoring` (for
        both training and validation datasets), and :class:`.PrintLog`
        callbacks are installed for the user's convenience.""",
        """    By default several logging and reporting callbacks for both Skorch
        and Ray Train are added to provide necessary functionality and
        a basic output.""").replace(
            """    iterator_train : torch DataLoader
        The default PyTorch :class:`~torch.utils.data.DataLoader` used for
        training data.

        iterator_valid : torch DataLoader
        The default PyTorch :class:`~torch.utils.data.DataLoader` used for
        validation and test data, i.e. during inference.""",
            """    iterator_train : PipelineIterator
        Iterator over ray.data.DatasetPipeline with conversion to torch
        Tensors. Used for validation and test data, i.e. during inference.
        Setting ``iterator_train__feature_columns`` and
        ``iterator_train__feature_column_dtypes`` gives control over what
        features and datatypes are set. Those parameters can also take in
        a list of lists/dicts in case of multi-input modules.

        iterator_valid : torch PipelineIterator
        As in ``iterator_train``, but for validation data. Please note
        that it needs its parameters to be set separately.""")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__,
        _docstring_neural_net_ray_worker_dataset,
        "    train_split : None or callable")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__, _docstring_neural_net_ray_args,
        "    optimizer : torch optim")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__,
        _docstring_neural_net_ray_train_callbacks,
        "    predict_nonlinearity : callable")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__, _docstring_neural_net_ray_trainer,
        "    Attributes")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__, _docstring_neural_net_ray_kwargs,
        "    Attributes")
    ray_train_neural_net_class.__doc__ = insert_before_substring(
        ray_train_neural_net_class.__doc__,
        _docstring_neural_net_ray_ray_train_history,
        "    _modules : list of str")

    ray_train_neural_net_class.fit.__doc__ = NeuralNet.fit.__doc__.replace(
        " (unless ``warm_start`` is True).", "").replace(
            """            * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
                * scipy sparse CSR matrices
                * a dictionary of the former three
                * a list/tuple of the former three
                * a Dataset""", """            * numpy arrays
                * pandas DataFrame or Series
                * a dictionary of the former two
                * a list/tuple of the former two
                * a ray.data.Dataset
                * a ray.data.DatasetPipeline
                * a Dataset

            All inputs will be automatically converted to ray.data
            formats internally.""")

    ray_train_neural_net_class.fit.__doc__ = insert_before_substring(
        ray_train_neural_net_class.fit.__doc__,
        _docstring_neural_net_ray_fit_params, "        **fit_params : dict")


def set_worker_neural_net_docs(worker_neural_net_class):
    worker_neural_net_class.__doc__ = NeuralNet.__doc__.replace(
        "NeuralNet base class.",
        "Internal use only. NeuralNet used inside each Ray Train worker.")
    worker_neural_net_class.__doc__ = insert_before_substring(
        worker_neural_net_class.__doc__, _docstring_neural_net_ray_kwargs,
        "    Attributes")
