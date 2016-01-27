## SGD.py    Dana Hughes    version 1.1     03-Sept-2015
##
## Stochastic Gradient Descent trainer for an arbitrary model.  This trainer 
## takes a model and a dataset and trains the model.  Make a better description.
##
## This trainer trains on batches using gradient descent 
##
## History:
##   1.0	21-Feb-2015	Initial version, modified algorithms from training.py
##   1.1	03-Sept-2015	Modified to work with new Neural Network code

import numpy as np
import random
import copy


class SGDTrainer(object):
   """
   A trainer which uses stochastic gradient descent.
   """

   def __init__(self, model, **kwargs):
      """
      Create a new SGD trainer.
      """

      self.model = model
      self.logger = kwargs.get('logger', None)
      self.learning_rate = kwargs.get('learning_rate', 0.01)
      self.momentum = kwargs.get('momentum', 0.0)
      self.weight_decay = kwargs.get('weight_decay', 0.0)

      self.prior_updates = None       # For momentum 


   def trainBatch(self, dataset):
      """
      Train once on each of the items in the provided batch
      """

      # Calculate the gradient of the cost function of the model given the data
      gradients = self.model.gradient(dataset, True)
      
      # Was there a prior weight update (e.g., for momentum)?
      if self.prior_updates == None:
         # Initialize prior gradients to zero
         self.prior_updates = np.zeros(gradients.shape)

      # What's the current parameters
      current_parameters = self.model.getFlatParameters()

      # Calculate the gradient updates
      updates = np.zeros(gradients.shape)

      # Gradient Descent
      updates = -self.learning_rate * gradients

      # Momentum
      if self.momentum > 0.0:
         updates -= self.momentum * self.prior_updates 

      # Weight Decay
      if self.weight_decay > 0.0:
         updates -= self.weight_decay * current_parameters

      # Store the current updates for use in the next step
      self.prior_updates = updates[:]

      # Update the model parameters
      self.model.updateFlatParameters(updates)


   def train(self, batchGenerator, **kwargs):
      """
      General training method.  Each training epoch, batches are produced using
      the batchGenerator until exhausted.
      """

      # Optional arguments - an object which indicates when to stop, and
      # a rate update ruleset
      stoppingCriteria = kwargs.get('stopping_criteria', None)
      rateUpdate = kwargs.get('rate_update', None)

      
