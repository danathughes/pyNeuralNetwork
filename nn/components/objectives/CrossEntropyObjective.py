## CrossEntropyObjective.py
##
## An objective layer which calculates the cross entropy between predicted 
## values and a target value.  Usually used in conjunction with softmax
## layers for classification purposes

import numpy as np

class CrossEntropyObjective:
   """
   Objective layer which calculates the cross entropy error between 
   predictions and targets
   """

   def __init__(self):
      """
      Create a new instance of a cross entropy objective.
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


      self.objective = -np.sum(self.targetLayer.getOutput() * np.log(self.outputLayer.getOutput()))
      self.objective += -np.sum((1.0 - self.targetLayer.getOutput())*(np.log(1.0 - self.outputLayer.getOutput())))


   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      self.delta = (self.targetLayer.getOutput() - self.outputLayer.getOutput())
      self.delta /= (self.outputLayer.getOutput() * (1.0 - self.outputLayer.getOutput()))


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
