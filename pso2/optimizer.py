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

    def print_particle(self):
        print('particle_{}: {}'.format(self.identifier, self.position_i))


class PSO:
    def __init__(self, **params):
        self.c1 = params.get('c1')
        self.c2 = params.get('c2')
        self.w = params.get('w')
        self.max_iter = params.get('max_iter')
        self.n_pop = params.get('n_pop')
        self.boundaries = params.get('boundaries')

        self.err_best_g = -1        # Best error for group
        self.pos_gest_g = []        # Best position for group

        # Establish the swarm
        self.swarm = []
        for i in range(0, self.n_pop):
            p = self.generate_params_model()
            particle = Particle(p, i)
            self.swarm.append(particle)

    def generate_params_model(self):
        p = []

        for i in self.boundaries.keys():
            low = self.boundaries[i][0]
            high = self.boundaries[i][1] + 1
            randint = np.random.randint(low, high)
            p.append(randint)

        return p
