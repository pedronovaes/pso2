import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

from pso2.optimizer import PSO


def load_datasets():
    X_train = pd.read_csv('datasets/x_train.csv')
    X_test = pd.read_csv('datasets/x_test.csv')
    y_train = pd.read_csv('datasets/y_train.csv')
    y_test = pd.read_csv('datasets/y_test.csv')

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    init = time.time()

    # Load and dataprep
    X_train, X_test, y_train, y_test = load_datasets()

    # Model
    model = GradientBoostingRegressor()

    # Boundaries
    boundaries = [
        (10, 200),  # n_estimators
        (3, 15),    # max_depth
    ]

    # PSO params
    c1 = 1.0
    c2 = 2.0
    w = 0.5
    n_pop = 10
    max_iter = 2

    # Compress all parameters into a dict
    pso_params = {
        'model': model,
        'boundaries': boundaries,
        'c1': c1,
        'c2': c2,
        'w': w,
        'n_pop': n_pop,
        'max_iter': max_iter
    }

    opt = PSO(**pso_params)

    end = time.time()
    print('total time: {:.2f}'.format(end - init))
