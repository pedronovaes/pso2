import numpy as np
from pso2.models import set_model


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

        # Machine Learning model
        self.model = set_model(params.get('model'), params.get('boundaries'))

        # Establish the swarm
        self.swarm = []
        for i in range(0, self.n_pop):
            p = self.model.generate_params_model()
            print(p)
            particle = Particle(p, i)
            self.swarm.append(particle)

    def optimize(self):
        pass
