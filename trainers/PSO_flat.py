## PSO.py    Dana Hughes    version 1.0     25-Jan-2016
##
## Particle Swarm Optimization trainer for an arbitrary model.  This trainer 
## takes a model and a dataset and trains the model.  Make a better description.
##
## This trainer trains on batches using a PSO algorithm 
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
      self.w0 = kwargs.get('w0', 1.0)
      self.w_inf = kwargs.get('w_inf', 0.0)
      self.t = 0.0
      self.K = 100.0

      # Create the initial population
      self.positions = []
      self.velocities = []
      self.personal_bests = []
      self.personal_best_positions = []

      self.global_best = 1.0e20
      self.global_best_position = None

      # Initialize the population
      network_parameters = self.model.getFlatParameters()

      lo, hi = self.initial_weight_range

      for i in range(self.number_particles):
         self.positions.append(network_parameters.copy() + np.random.uniform(lo, hi, network_parameters.shape))
         self.velocities.append(np.random.uniform(lo, hi, network_parameters.shape))

         # These are just absurdly high values for initialization purposes
         self.personal_bests.append(1.0e20)
         self.personal_best_positions.append(self.positions[i].copy())


   def trainBatch(self, dataset):
      """
      Perform a single iteration of PSO
      """

      # Calculate the current fitness values for each particle
      for i in range(self.number_particles):
         # Set the model parameters to these parameters
         self.model.setFlatParameters(self.positions[i])
         self.model.evaluate(dataset)

         # Determine the objective, and replace the personal best if necessary
         objective = self.model.getObjective()
         if objective < self.personal_bests[i]:
            self.personal_bests[i] = objective
            self.personal_best_positions[i] = self.positions[i].copy()
       
         # Is this better than the global best?
         if objective < self.global_best:
            self.global_best = objective
            self.global_best_position = self.positions[i].copy()

      # Update positions and velocities
      for i in range(self.number_particles):
         c1_val = self.c1*random.random()
         c2_val = self.c2*random.random()

         self.velocities[i] *= self.w_inf + (self.w0 - self.w_inf)*(1.0 - self.t/self.K)
         self.velocities[i] += c1_val*(self.personal_best_positions[i] - self.positions[i])
         self.velocities[i] += c2_val*(self.global_best_position - self.positions[i])

         # Cap the velocities to max_velocity
         self.velocities[self.velocities > self.max_velocity] = self.max_velocity
         self.velocities[self.velocities < -self.max_velocity] = -self.max_velocity    

         self.positions[i] += self.velocities[i]

      # Finally, set the global best to the model
      self.model.setFlatParameters(self.global_best_position)
      self.t += 1.0



   def train(self, batchGenerator, **kwargs):
      """
      General training method.  Each training epoch, batches are produced using
      the batchGenerator until exhausted.
      """

      # Optional arguments - an object which indicates when to stop, and
      # a rate update ruleset
      stoppingCriteria = kwargs.get('stopping_criteria', None)
      rateUpdate = kwargs.get('rate_update', None)

      
