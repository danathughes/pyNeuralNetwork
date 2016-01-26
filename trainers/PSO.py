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

      self.global_best = []
      self.global_best_positions = []

      # Create the initial population
      


   def trainBatch(self, dataset):
      """
      Train once on each of the items in the provided batch
      """

      # Calculate the gradient of the cost function of the model given the data
      gradients = self.model.gradient(dataset)
      
      # Was there a prior weight update (e.g., for momentum)?
      if self.prior_updates == None:
         # Initialize prior gradients to zero
         self.prior_updates = {}
         for unit, grad in gradients.items():
            self.prior_updates[unit] = np.zeros(grad.shape)

      # What's the current parameters
      current_parameters = self.model.getParameters()

      # Calculate the gradient updates
      updates = {}
      for unit in gradients.keys():
         # Gradient Descent
         updates[unit] = -self.learning_rate * gradients[unit] 

         # Momentum
         if self.momentum > 0.0:
            updates[unit] -= self.momentum * self.prior_updates[unit] 

         # Weight Decay
         if self.weight_decay > 0.0:
            updates[unit] -= self.weight_decay * current_parameters[unit] 

         # Store the current updates for use in the next step
         self.prior_updates[unit] = updates[unit][:]

      # Update the model parameters
      self.model.updateParameters(updates)


   def train(self, batchGenerator, **kwargs):
      """
      General training method.  Each training epoch, batches are produced using
      the batchGenerator until exhausted.
      """

      # Optional arguments - an object which indicates when to stop, and
      # a rate update ruleset
      stoppingCriteria = kwargs.get('stopping_criteria', None)
      rateUpdate = kwargs.get('rate_update', None)

      
