## MSEObjective.py
##
##

import numpy as np

class MSEObjective:
   """
   """

   def __init__(self):
      """
      Create a new instance of an MSE objective.
      """

      self.outputLayer = None
      self.targetLayer = None
      self.objective = None
      self.delta = None

   def setOutputLayer(self, layer):
      """
      Connect the objective to its output layer.
      """

      self.outputLayer = layer


   def setTargetLayer(self, layer):
      """
      Connect the objective to its target layer.
      """

      self.targetLayer = layer


   def getObjective(self):
      """
      Get the current objective value.
      """

      return self.objective


   def forward(self):
      """
      Perform a forward pass to calculate the activation (objective)
      """

      self.objective = 0.5 * np.sum((self.targetLayer.getOutput() - self.outputLayer.getOutput())**2)


   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      self.delta = self.targetLayer.getOutput() - self.outputLayer.getOutput()


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

      return self.delta
