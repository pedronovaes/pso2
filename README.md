# pso2

Particle Swarm Optimization Optimizer (pso2) is a tool that uses PSO method as heuristics to find the best hyperparameters for Machine Learning algorithms in Python.

### what is Particle Swarm Optimization

PSO is a biologically inspired computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions (particles) and moving these particles around in the search-space according to simple mathematical formulas over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but it is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

### how can I use Particle Swarm Optimization to optimize Machine Learning model hyperparameters

Each particle is a Machine Learning model and the measure of quality is the model loss function, that is a pre-defined metric, such as mean squared error (MSE) and accuracy (ACC), for regression and classification problems, respectively.

### accepted models:

You can use pso2 to optimize the hyperparameters of the following sklearn-based models:

| Model | Hyperparameters |
| ------ | ------ |
| GradientBoostingRegressor | n_estimators, max_depth, learning_rate |

### how to use pso2

To use pso2, it is necessary to encapsulate some parameters in a dictionary. The parameters are: chosen model, model hyperparameters and its boundaries, loss function, training and test data, PSO parameters, number of particles, and number of iterations.

Let's demonstrate an example of Gradient Boosting Regression hyperparameters optiimization:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Model
model = GradientBoostingRegressor()

# Boundaries
boundaries = {
    'n_estimators': (10, 200),
    'max_depth': (3, 15),
    'learning_rate': (0.001, 0.1)
}

# Function to minimize (loss function)
loss_func = mean_squared_error

# PSO params
c1 = 1.0
c2 = 2.0
w = 0.5
n_pop = 10
max_iter = 10
```

You must encapsulate everything in a dict:

```python
pso_params = {
    'model': model,
    'boundaries': boundaries,
    'loss_func': loss_func,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'c1': c1,
    'c2': c2,
    'w': w,
    'n_pop': n_pop,
    'max_iter': max_iter,
    'verbose': 1,
}
```

And instantiate a PSO object:

```python
from pso2.optimizer import PSO

opt = PSO(**pso_params)
opt.optimize()
```

You can access the best model hyperparameters using the **best_params_** attribute:

```python
print(opt.best_params_)
```
