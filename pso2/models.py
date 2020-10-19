import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GBR:
    def __init__(self, params):
        self.params = params

    def generate_params_model(self):
        p = []

        for i in self.params.keys():
            low = self.params[i][0]
            high = self.params[i][1] + 1
            num = np.random.randint(low, high)
            p.append(num)

        return p


def set_model(m, params):
    if isinstance(m, GradientBoostingRegressor):
        model = GBR(params)

    return model
