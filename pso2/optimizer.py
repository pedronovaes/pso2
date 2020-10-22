import copy
import numpy as np
from pso2.models import set_model


class Particle:
    def __init__(self, parameters, identifier):
        self.identifier = identifier            # Particle identifier
        self.position_i = parameters            # Particle position
        self.velocity_i = []                    # Particle velocity
        self.pos_best_i = []                    # Best position individual
        self.err_best_i = -1                    # Best error individual
        self.err_i = -1                         # Error individual
        self.num_dimensions = len(parameters)   # Num of problem dimensions

        # Generating initial velocities randomly
        for i in range(0, self.num_dimensions):
            v = np.random.rand(1)[0]
            self.velocity_i.append(v)

    def print_particle(self):
        print('particle_{}: {}'.format(self.identifier, self.position_i))

    def print_loss(self):
        print('loss_{}: {}'.format(self.identifier, self.err_i))

    # Evaluate current fitness
    def evaluate(self, model):
        self.err_i = model.fit(self.position_i)

        # Check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = copy.deepcopy(self.position_i)
            self.err_best_i = self.err_i


class PSO:
    def __init__(self, **params):
        self.c1 = params.get('c1')
        self.c2 = params.get('c2')
        self.w = params.get('w')
        self.max_iter = params.get('max_iter')
        self.n_pop = params.get('n_pop')
        self.boundaries = params.get('boundaries')
        self.loss_func = params.get('loss_func')

        self.err_best_g = -1        # Best error for group
        self.pos_gest_g = []        # Best position for group

        # Machine Learning model
        self.model = set_model(
            m=params.get('model'),
            params=params.get('boundaries'),
            X_train=params.get('X_train'),
            X_test=params.get('X_test'),
            y_train=params.get('y_train'),
            y_test=params.get('y_test'),
            loss_func=params.get('loss_func')
        )

        # Establish the swarm
        self.swarm = []
        for i in range(0, self.n_pop):
            p = self.model.generate_params_model()
            particle = Particle(p, i)
            self.swarm.append(particle)

    def optimize(self):
        i = 0

        # Begin optimization loop
        while i < self.max_iter:
            # Cycle through particles in swarm and evaluate fitness
            for j in range(0, self.n_pop):
                self.swarm[j].evaluate(self.model)
                self.swarm[j].print_loss()
                self.swarm[j].print_particle()

                # Determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_gest_g = copy.deepcopy(self.swarm[j].position_i)
                    self.err_best_g = self.swarm[j].err_i

            i += 1
