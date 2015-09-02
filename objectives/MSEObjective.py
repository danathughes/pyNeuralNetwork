## MSEObjective.py
##
##

import numpy as np

class MSEObjective:
   """
   """

   def __init__(self):
      """
      """

      self.outputLayer = None
      self.targetLayer = None
      self.gradient = None
      self.objective = None

   def setOutputLayer(self, layer):
      """
      """

      self.outputLayer = layer


   def setTargetLayer(self, layer):
      """
      """

      self.targetLayer = layer


   def getObjective(self):
      """
      """

      return self.objective


   def forward(self):
      """
      """

      self.objective = 0.5 * np.sum((self.targetLayer.getOutput() - self.outputLayer.getOutput())**2)


   def backward(self):
      """
      """

      self.gradient = self.targetLayer.getOutput() - self.outputLayer.getOutput()


   def getGradient(self):
      """
      """

      return self.gradient
