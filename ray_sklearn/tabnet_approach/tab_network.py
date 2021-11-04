import numpy as np
import torch
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from pytorch_tabnet.utils import filter_weights

from sklearn.base import RegressorMixin, ClassifierMixin

from ray_sklearn.tabnet_approach.abstract_model import TabModel


class TabNetClassifier(TabModel, ClassifierMixin):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = "accuracy"

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn_(y_pred, y_true.long())

    def update_fit_params(
            self,
            X_train,
            y_train,
            eval_set,
            weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim_ = output_dim
        self._default_metric = "auc" if self.output_dim_ == 2 else "accuracy"
        self.classes_ = train_labels
        self.target_mapper_ = {
            class_label: index
            for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper_ = {
            str(index): class_label
            for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights_ = self.weight_updater(weights)


class TabNetRegressor(TabModel, RegressorMixin):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = "regression"
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = "mse"

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn_(y_pred, y_true)

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        if len(y_train.shape) != 2:
            msg = ("Targets should be 2D : (n_samples, n_regression) " +
                   f"but y_train.shape={y_train.shape} given.\n" +
                   "Use reshape(-1, 1) for single regression.")
            raise ValueError(msg)
        self.output_dim_ = y_train.shape[1]
        self.preds_mapper_ = None

        self.updated_weights_ = weights
        filter_weights(self.updated_weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
