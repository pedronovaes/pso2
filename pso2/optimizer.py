import time
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
        self.model_i = None                     # Model individual
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
        self.err_i, self.model_i = model.fit(self.position_i)

        # Check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = copy.deepcopy(self.position_i)
            self.err_best_i = self.err_i

    # Update new particle velocity
    def update_velocity(self, pos_best_g, c1, c2, w):
        """
        w: Constant inertia weight (how much to weigh the previous velocity)
        c1: Cognitive constant
        c2: Social constant
        """

        for i in range(0, self.num_dimensions):
            r1 = np.random.rand(1)[0]
            r2 = np.random.rand(1)[0]

            cognitive_vel = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            social_vel = c2 * r2 * (pos_best_g[i] - self.position_i[i])

            self.velocity_i[i] = w * self.velocity_i[i] + cognitive_vel + social_vel

    # Update the particle position based off new velocity updates
    def update_position(self, boundaries):
        for i in range(0, self.num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # Adjust maximum position if necessary
            if self.position_i[i] > boundaries[i][1]:
                self.position_i[i] = boundaries[i][1]

            # Adjsut minimum position if necessary
            if self.position_i[i] < boundaries[i][0]:
                self.position_i[i] = boundaries[i][0]


class PSO:
    def __init__(self, **params):
        self.c1 = params.get('c1')
        self.c2 = params.get('c2')
        self.w = params.get('w')
        self.max_iter = params.get('max_iter')
        self.n_pop = params.get('n_pop')
        self.boundaries = params.get('boundaries')
        self.loss_func = params.get('loss_func')
        self.verbose = params.get('verbose')

        self.err_best_g = -1        # Best error for group
        self.pos_best_g = []        # Best position for group
        self.model_best_g = None    # Best model for group
        self.best_params_ = {}

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
        init = time.time()

        i = 0

        # Begin optimization loop
        while i < self.max_iter:
            if self.verbose:
                print('iter_{}'.format(i))

            # Cycle through particles in swarm and evaluate fitness
            for j in range(0, self.n_pop):
                self.swarm[j].evaluate(self.model)

                if self.verbose == 2:
                    self.swarm[j].print_loss()
                    self.swarm[j].print_particle()

                # Determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = copy.deepcopy(self.swarm[j].position_i)
                    self.err_best_g = self.swarm[j].err_i
                    self.model_best_g = copy.deepcopy(self.swarm[j].model_i)

            # Cycle through swarm and update velocities and positions
            for j in range(0, self.n_pop):
                self.swarm[j].update_velocity(self.pos_best_g, self.c1, self.c2, self.w)
                self.swarm[j].update_position(list(self.boundaries.values()))

            if self.verbose:
                print('best loss: {}'.format(self.err_best_g))

            i += 1

        end = time.time()

        if self.verbose:
            print('total time: {:.2f}s'.format(end - init))

        self.best_params_ = self.format_output()

        return self.model_best_g

    def format_output(self):
        output = {}

        for i, key in enumerate(self.boundaries.keys()):
            output[key] = self.model.get_types(key)(self.pos_best_g[i])

        return output
