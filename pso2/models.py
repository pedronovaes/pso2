import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC


# Base class of models
class Model:
    def __init__(self, params, X_train, X_test, y_train, y_test, loss_func, model):
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss_func = loss_func
        self.model = model

    # Method to get hyperparam type
    def get_types(self, param):
        pass

    # Given a key and a value, this methods returns a random value of a
    # specific hyperparam
    def get_params(self, key, value):
        pass

    # Generates initial model hyperparams
    def generate_params_model(self):
        p = []

        for i in self.params.keys():
            num = self.get_params(i, self.params[i])
            p.append(num)

        return p

    # Train a model and returns some loss function value
    def fit(self, params):
        params_dict = {}

        for i, key in enumerate(self.params.keys()):
            params_dict[key] = self.get_types(key)(params[i])

        model = self.model(**params_dict)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        loss = self.loss_func(self.y_test, y_pred)

        return loss, model


class GradientBoosting(Model):
    def get_types(self, param):
        types = {
            'n_estimators': np.int64,
            'max_depth': np.int64,
            'learning_rate': np.float64,
            'tol': np.float64,
        }

        return types[param]

    def get_params(self, key, value):
        low = value[0]
        high = value[1]

        params = {
            'n_estimators': np.random.randint(low, high + 1),
            'max_depth': np.random.randint(low, high + 1),
            'learning_rate': np.random.uniform(low, high),
            'tol': np.random.uniform(low, high),
        }

        return params[key]


class SupportVectorMachines(Model):
    def get_types(self, param):
        types = {
            'C': np.float64,
            'tol': np.float64
        }

        return types[param]

    def get_params(self, key, value):
        low = value[0]
        high = value[1]

        params = {
            'C': np.random.uniform(low, high),
            'tol': np.random.uniform(low, high),
        }

        return params[key]


def set_model(m, params, X_train, X_test, y_train, y_test, loss_func):
    if isinstance(m, GradientBoostingRegressor):
        model = GradientBoosting(params, X_train, X_test, y_train, y_test, loss_func, GradientBoostingRegressor)
    elif isinstance(m, GradientBoostingClassifier):
        model = GradientBoosting(params, X_train, X_test, y_train, y_test, loss_func, GradientBoostingClassifier)
    elif isinstance(m, SVR):
        model = SupportVectorMachines(params, X_train, X_test, y_train, y_test, loss_func, SVR)
    elif isinstance(m, SVC):
        model = SupportVectorMachines(params, X_train, X_test, y_train, y_test, loss_func, SVC)

    return model
