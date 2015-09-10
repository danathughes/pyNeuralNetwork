## MultiObjective.py		Dana Hughes		08-Sept-2015
##
## An objective which is simply the weighted sum of multiple objectives.  To use
## this properly, it should be treated as a container of other objectives.  
## Objectives should be added to this container, then this objective (alone) should 
## be added to the neural network
##
## History:
##	1.0	08-Sept-2015	Initial Version

import numpy as np

class MultiObjective:
   """
   An objective which simply combines the weighted sum of multiple objectives
   """

   def __init__(self):
      """
      Create a new multiple objective.
      """

      # Maintain a parallel list of objectives and weights
      self.objectives = []
      self.weights = []
 
      # The objective will simply be the sum of all objectives 
      # in the list
      self.objective = None


   def addObjective(self, objective, weight = 1.0):
      """
      Add the objective and weight to the list of objectives and weights
      """

      self.objectives.append(objective)
      self.weights.append(weight)


   def getObjective(self):
      """
      Get the current objective value.
      """

      return self.objective


   def forward(self):
      """
      Perform a forward pass to calculate the activation (objective)
      """

      # Perform a forward pass on each of the objectives
      for objective in self.objectives:
         objective.forward()

      # Compute the weighted sum of all the objectives
      self.objective = sum([weight*obj.getObjective() for weight, obj in zip(self.weights, self.objectives)])


   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      # Perform a backward pass on each objective, then multiply the deltas
      # by the weights associated with each delta
      for weight, objective in zip(self.weights, self.objectives):
         objective.backward()
         objective.delta *= weight


   def getParameterGradient(self):
      """
      Return the gradient of the parameters for this module
      """

      # This objective function has no parameters
      return None


   def getDelta(self):
      """
      Return the delta after backward
      """

      return None
