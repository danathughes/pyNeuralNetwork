## PSO.py    Dana Hughes    version 1.0     25-Jan-2016
##
## Particle Swarm Optimization trainer for an arbitrary model.  This trainer 
## takes a model and a dataset and trains the model.  Make a better description.
##
## This trainer trains on batches using gradient descent 
##
## History:
##   1.0	25-Jan-2016	Initial version, modified skeleton from SGD.py

import numpy as np
import random
import copy


class PSOTrainer(object):
   """
   A trainer which uses particle swarm optimization.

   NOTE:  The initial_weight_range is the range of the random weight to *add* 
          to the existing model parameters, not *replace* model parameters.
   """

   def __init__(self, model, **kwargs):
      """
      Create a new PSO trainer.
      """

      self.model = model
      self.logger = kwargs.get('logger', None)

      # Parameters for the PSO algorithm
      self.number_particles = kwargs.get('number_particles', 10)
      self.max_velocity = kwargs.get('max_velocity', 1.0)
      self.initial_weight_range = kwargs.get('initial_weight_range', (-1.0, 1.0))
      self.c1 = kwargs.get('c1', 2.0)
      self.c2 = kwargs.get('c2', 2.0)

      # Create the initial population
      self.positions = []
      self.velocities = []
      self.personal_bests = []
      self.personal_best_positions = []

      self.global_best = 1.0e20
      self.global_best_position = []

      model_params = self.model.getParameters()
      self.parameter_keys = model_params.keys()

      # Create the initial population
      for i in range(self.number_particles):
         # Copy the model parameters into a list and add random noise to the position
         parameters = [model_params[key].copy() for key in self.parameter_keys]
         parameters = [param + np.random.uniform(self.initial_weight_range[0], self.initial_weight_range[1], param.shape) for param in parameters]
         self.positions.append(parameters)

         # Set up a 'dummy' personal best
         self.personal_bests.append(1.0e20)
         self.personal_best_positions.append(parameters)

         velocity = [0.0*param for param in parameters]
         self.velocities.append(velocity)


   def trainBatch(self, dataset):
      """
      Perform a single iteration of PSO
      """

      # Calculate the current fitness values for each particle
      for i in range(self.number_particles):
         # Write particle i's parameters to the model and get the fitness
         params = {}
         for key, param in zip(self.parameter_keys, self.positions[i]):
            params[key] = param
         self.model.setParameters(params)

         # Calculating the gradient performs all the stuff
         self.model.gradient(dataset)
         objective = self.model.getObjective()
        
         # is this a new personal best?
         if objective < self.personal_bests[i]:
            self.personal_bests[i] = objective
            self.personal_best_positions[i] = [pos.copy() for pos in self.positions[i]]

      # Update the global best
      for i in range(self.number_particles):
         if self.personal_bests[i] < self.global_best:
            self.global_best = self.personal_bests[i]
            self.global_best_position = [pos.copy() for pos in self.personal_best_positions[i]]

      # Update positions and velocities
      for i in range(self.number_particles):
         c1_val = self.c1*random.random()
         c2_val = self.c2*random.random()

         for j in range(len(self.velocities[i])):
            self.velocities[i][j] += c1_val*(self.personal_best_positions[i][j] - self.positions[i][j])
            self.velocities[i][j] += c2_val*(self.global_best_position[j] - self.positions[i][j])

            # So, gotta implement max_velocity here somehow...

            # Update positions
            self.positions[i][j] += self.velocities[i][j]

      # Finally, set the global best to the model
      params = {}
      for key, param in zip(self.parameter_keys, self.global_best_position):
         params[key] = param
      self.model.setParameters(params)


   def train(self, batchGenerator, **kwargs):
      """
      General training method.  Each training epoch, batches are produced using
      the batchGenerator until exhausted.
      """

      # Optional arguments - an object which indicates when to stop, and
      # a rate update ruleset
      stoppingCriteria = kwargs.get('stopping_criteria', None)
      rateUpdate = kwargs.get('rate_update', None)

      
