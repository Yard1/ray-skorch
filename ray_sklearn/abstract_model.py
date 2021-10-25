import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Dict

import numpy as np
import torch
from pytorch_tabnet.callbacks import CallbackContainer
from pytorch_tabnet.metrics import MetricContainer, check_metrics
from pytorch_tabnet.utils import (
    validate_eval_set,
    create_dataloaders,
    define_device,
)
from ray import train
from ray.train import Trainer
from ray_sklearn.models import tabnet as tab_network
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from torch.nn.parallel import DistributedDataParallel


@dataclass
class TabModel(BaseEstimator):
    """ Class for TabNet model."""

    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    cat_emb_dim: int = 1
    n_independent: int = 2
    n_shared: int = 2
    epsilon: float = 1e-15
    momentum: float = 0.02
    lambda_sparse: float = 1e-3
    seed: int = 0
    clip_value: int = 1
    verbose: int = 1
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2))
    scheduler_fn: Any = None
    scheduler_params: Dict = field(default_factory=dict)
    mask_type: str = "sparsemax"
    input_dim: int = None
    output_dim: int = None
    device_name: str = "auto"
    n_shared_decoder: int = 1
    n_indep_decoder: int = 1

    def __post_init__(self):
        self.batch_size = 1024
        self.virtual_batch_size = 128
        torch.manual_seed(self.seed)
        # Defining device
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            print(f"Device used : {self.device}")

    def __update__(self, **kwargs):
        """
        Updates parameters.
        If does not already exists, creates it.
        Otherwise overwrite with warnings.
        """
        update_list = [
            "cat_dims",
            "cat_emb_dim",
            "cat_idxs",
            "input_dim",
            "mask_type",
            "n_a",
            "n_d",
            "n_independent",
            "n_shared",
            "n_steps",
        ]
        for var_name, value in kwargs.items():
            if var_name in update_list:
                try:
                    exec(
                        f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # noqa
                        wrn_msg = f"Pretraining: {var_name} changed from {previous_val} to {value}"  # noqa
                        warnings.warn(wrn_msg)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

    def fit(
            self,
            X_train,
            y_train,
            eval_set=None,
            eval_name=None,
            eval_metric=None,
            loss_fn=None,
            weights=0,
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=None,
            pin_memory=True,
            # from_unsupervised=None,
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
            dict for custom weights per class
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        from_unsupervised: unsupervised trained model
            Use a previously self supervised model as starting weights
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        # self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_array(X_train)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train,
                                                 y_train)

        # TODO convert to ray datasets
        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set)

        # if from_unsupervised is not None:
        #     # Update parameters to match self pretraining
        #     self.__update__(**from_unsupervised.get_params())

        if not hasattr(self, "network"):
            self._set_network_creator()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer_creator()
        self._set_callbacks(callbacks)

        trainer = Trainer("torch", 2)
        train_func = self._generate_train_func(train_dataloader,
                                               valid_dataloaders, eval_names)
        trainer.start()
        trainer.run(train_func)
        trainer.shutdown()

    def _generate_train_func(self, train_dataloader, valid_dataloaders,
                             eval_names):
        def train_epoch(network, device, optimizer, train_loader):
            """
            Trains one epoch of the network in self.network

            Parameters
            ----------
            train_loader : a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
            """
            network.train()
            all_logs = []

            for batch_idx, (X, y) in enumerate(train_loader):
                # self._callback_container.on_batch_begin(batch_idx)

                batch_logs = train_batch(network, device, optimizer, X, y)
                all_logs.append(batch_logs)

                # self._callback_container.on_batch_end(batch_idx, batch_logs)

            # epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
            # self.history.epoch_metrics.update(epoch_logs)
            return all_logs

        def train_batch(network, device, optimizer, X, y):
            """
            Trains one batch of data

            Parameters
            ----------
            X : torch.Tensor
                Train matrix
            y : torch.Tensor
                Target matrix

            Returns
            -------
            batch_outs : dict
                Dictionnary with "y": target and "score": prediction scores.
            batch_logs : dict
                Dictionnary with "batch_size" and "loss".
            """
            batch_logs = {"batch_size": X.shape[0]}

            X = X.to(device).float()
            y = y.to(device).float()

            for param in network.parameters():
                param.grad = None

            output, M_loss = network(X)

            loss = self.compute_loss(output, y)
            # Add the overall sparsity loss
            loss -= self.lambda_sparse * M_loss

            # Perform backward pass and optimization
            loss.backward()
            # if self.clip_value:
            #     clip_grad_norm_(self.network.parameters(), self.clip_value)
            optimizer.step()

            batch_logs["loss"] = loss.cpu().detach().numpy().item()

            return batch_logs

        def predict_epoch(network, device, name, loader):
            """
            Predict an epoch and update metrics.

            Parameters
            ----------
            name : str
                Name of the validation set
            loader : torch.utils.data.Dataloader
                    DataLoader with validation set
            """
            # Setting network on evaluation mode
            network.eval()

            list_y_true = []
            list_y_score = []

            # Main loop
            for batch_idx, (X, y) in enumerate(loader):
                scores = predict_batch(network, device, X)
                list_y_true.append(y)
                list_y_score.append(scores)

            y_true, scores = self.stack_batches(list_y_true, list_y_score)

            # metrics_logs = self._metric_container_dict[name](y_true, scores)
            network.train()
            # self.history.epoch_metrics.update(metrics_logs)
            return (y_true, scores)

        def predict_batch(network, device, X):
            """
            Predict one batch of data.

            Parameters
            ----------
            X : torch.Tensor
                Owned products

            Returns
            -------
            np.array
                model scores
            """
            X = X.to(device).float()

            # compute model output
            scores, _ = network(X)

            if isinstance(scores, list):
                scores = [x.cpu().detach().numpy() for x in scores]
            else:
                scores = scores.cpu().detach().numpy()

            return scores

        def train_func():
            # TODO: This should not require serializing the entire TabModel.
            use_gpu = False  # TODO

            network = self._network_creator()
            device = torch.device(f"cuda:{train.local_rank()}" if use_gpu
                                  and torch.cuda.is_available() else "cpu")

            network.to(device)
            network = DistributedDataParallel(
                network, find_unused_parameters=True)
            optimizer = self._optimizer_creator(network)

            # # Call method on_train_begin for all callbacks
            # self._callback_container.on_train_begin()

            # Training loop over epochs
            for epoch_idx in range(self.max_epochs):

                # # Call method on_epoch_begin for all callbacks
                # self._callback_container.on_epoch_begin(epoch_idx)

                r = train_epoch(network, device, optimizer, train_dataloader)
                print(f"Epoch:[{epoch_idx}] Train results: {r}")

                # Apply predict epoch to all eval sets
                predict_results = []
                for eval_name, valid_dataloader in zip(eval_names,
                                                       valid_dataloaders):
                    r = predict_epoch(network, device, eval_name,
                                      valid_dataloader)
                    predict_results.append(r)
                # print(f"Predict results: {predict_results}")

            # Call method on_train_end for all callbacks
            # self._callback_container.on_train_end()
            network.eval()

            # # compute feature importance once the best model is defined
            # self._compute_feature_importances(train_dataloader)

        return train_func

    def load_class_attrs(self, class_attrs):
        for attr_name, attr_value in class_attrs.items():
            setattr(self, attr_name, attr_value)

    def _set_network_creator(self):
        """Setup the network and explain matrix."""

        def network_creator():
            return tab_network.TabNet(
                self.input_dim,
                self.output_dim,
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                cat_emb_dim=self.cat_emb_dim,
                n_independent=self.n_independent,
                n_shared=self.n_shared,
                epsilon=self.epsilon,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                mask_type=self.mask_type,
            )

        self._network_creator = network_creator

        # self.reducing_matrix = create_explain_matrix(
        #     self.network.input_dim,
        #     self.network..cat_emb_dim,
        #     self.network.cat_idxs,
        #     self.network.post_embed_dim, #TODO
        # )

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update({
                name: MetricContainer(metrics, prefix=f"{name}_")
            })

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (self._metrics_names[-1] if
                                      len(self._metrics_names) > 0 else None)

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        # self.history = History(self, verbose=self.verbose)
        # callbacks.append(self.history)
        # if (self.early_stopping_metric is not None) and (self.patience > 0):
        #     early_stopping = EarlyStopping(
        #         early_stopping_metric=self.early_stopping_metric,
        #         is_maximize=(
        #             self._metrics[-1]._maximize if len(self._metrics) > 0 else None
        #         ),
        #         patience=self.patience,
        #     )
        #     callbacks.append(early_stopping)
        # else:
        #     print(
        #         "No early stopping will be performed, last training weights will be used."
        #     )
        # if self.scheduler_fn is not None:
        #     # Add LR Scheduler call_back
        #     is_batch_level = self.scheduler_params.pop("is_batch_level", False)
        #     scheduler = LRSchedulerCallback(
        #         scheduler_fn=self.scheduler_fn,
        #         scheduler_params=self.scheduler_params,
        #         optimizer=self._optimizer,
        #         early_stopping_metric=self.early_stopping_metric,
        #         is_batch_level=is_batch_level,
        #     )
        #     callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer_creator(self):
        """Setup optimizer creator."""

        def optimizer_creator(network):
            return self.optimizer_fn(network.parameters(),
                                     **self.optimizer_params)

        self._optimizer_creator = optimizer_creator

    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders

    def _update_network_params(self):
        # TODO use this to change network, e.g. through `train_func(config)`
        pass
        # self.network.virtual_batch_size = self.virtual_batch_size

    @abstractmethod
    def update_fit_params(self, X_train, y_train, eval_set, weights):
        """
        Set attributes relative to fit function.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
        """
        raise NotImplementedError(
            "users must define update_fit_params to use this base class")

    @abstractmethod
    def compute_loss(self, y_score, y_true):
        """
        Compute the loss.

        Parameters
        ----------
        y_score : a :tensor: `torch.Tensor`
            Score matrix
        y_true : a :tensor: `torch.Tensor`
            Target matrix

        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError(
            "users must define compute_loss to use this base class")

    @abstractmethod
    def prepare_target(self, y):
        """
        Prepare target before training.

        Parameters
        ----------
        y : a :tensor: `torch.Tensor`
            Target matrix.

        Returns
        -------
        `torch.Tensor`
            Converted target matrix.
        """
        raise NotImplementedError(
            "users must define prepare_target to use this base class")
