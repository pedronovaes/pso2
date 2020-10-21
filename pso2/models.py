import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GBR:
    def __init__(self, params):
        self.params = params
        print(self.params)

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


def set_model(m, params):
    if isinstance(m, GradientBoostingRegressor):
        model = GBR(params)

    return model
