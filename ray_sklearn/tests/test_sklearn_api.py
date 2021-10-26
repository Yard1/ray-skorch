from typing import Optional
import pytest
from sklearn.utils.estimator_checks import check_estimator

from ray_sklearn.tabnet_approach import TabNetClassifier, TabNetRegressor

_estimator_classes = [TabNetClassifier, TabNetRegressor]


@pytest.mark.parametrize("estimator_class", _estimator_classes)
def test_check_estimator(estimator_class: type, params: Optional[dict] = None):
    params = {} or params
    check_estimator(estimator_class(**params))
