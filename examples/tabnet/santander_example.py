from ray_sklearn import TabNetRegressor
from ray_sklearn.data import load_santander

# https://www.kaggle.com/carlmcbrideellis/tabnet-simple-binary-classification-example/data

train_data, test_data = load_santander()

X_train = train_data.iloc[:, :-1].to_numpy()
y_train = train_data["TARGET"].to_numpy().squeeze()
X_test = test_data.to_numpy()

regressor = TabNetRegressor()

regressor.fit(
    X_train=X_train,
    y_train=y_train,
    patience=5,
    max_epochs=100,
    eval_metric=["auc"])

predictions = regressor.predict_proba(X_test)[:, 1]
