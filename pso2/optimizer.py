import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class Particle:
    def __init__(self, parameters, identifier):
        self.identifier = identifier    # Particle identifier
        self.position_i = parameters    # Particle position
        self.velocity_i = []            # Particle velocity
        self.pos_best_i = []            # Best position individual
        self.err_best_i = -1            # Best error individual
        self.err_i = -1                 # Error individual

        # Generating initial velocities randomly
        for i in range(0, num_dimensions):
            self.velocity_i.append(np.random.rand(1)[0])

        def print_particle(self):
            print('particle_{}: {}'.format(self.identifier, self.position_i))

        def print_loss(self):
            print('loss_{}: {}'.format(self.identifier, self.err_i))


class PSO:
    def __init__(self, **params):
        self.c1 = params.get('c1')
        self.c2 = params.get('c2')
        self.w = params.get('w')
        self.max_iter = params.get('max_iter')
        self.n_pop = params.get('n_pop')
        self.boundaries = params.get('boundaries')
        self.model = params.get('model')

        print(self.c1)
        print(self.c2)
        print(self.w)
        print(self.max_iter)
        print(self.n_pop)
        print(self.boundaries)
        print(self.model)
