import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GBR:
    def __init__(self, params, X_train, X_test, y_train, y_test, loss_func):
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss_func = loss_func

    def get_types(self, param):
        types = {
            'n_estimators': np.int64,
            'max_depth': np.int64,
        }

        return types[param]

    def get_params(self, key, value):
        low = value[0]
        high = value[1]

        params = {
            'n_estimators': np.random.randint(low, high + 1),
            'max_depth': np.random.randint(low, high + 1),
        }

        return params[key]

    def generate_params_model(self):
        p = []

        for i in self.params.keys():
            num = self.get_params(i, self.params[i])
            p.append(num)

        return p

    def fit(self, params):
        # params = dict(zip(self.params.keys(), params))
        params_dict = {}

        for i, key in enumerate(self.params.keys()):
            params_dict[key] = self.get_types(key)(params[i])

        model = GradientBoostingRegressor(**params_dict)
        model.fit(self.X_train, self.y_train.values.ravel())

        y_pred = model.predict(self.X_test)
        loss = self.loss_func(self.y_test, y_pred)

        return loss


def set_model(m, params, X_train, X_test, y_train, y_test, loss_func):
    if isinstance(m, GradientBoostingRegressor):
        model = GBR(params, X_train, X_test, y_train, y_test, loss_func)

    return model
