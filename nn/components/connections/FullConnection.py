## FullConnection.py	Dana Hughes		01-Sept-2015
##

import numpy as np
from AbstractConnection import AbstractConnection

class FullConnection(AbstractConnection):
   """
   A connection which fully links two layers.
   """

   def __init__(self, inputSize, outputSize):
      """
      Create a new full connection
      """

      # Properly initialize the abstract connection
      AbstractConnection.__init__(self)

      self.parameters = np.zeros((inputSize, outputSize))
      self.dimensions = (inputSize, outputSize)

      self.inputs = np.zeros((0,inputSize))
      self.outputs = np.zeros((0,outputSize))

      self.gradient = np.zeros(self.dimensions)


   def randomize(self):
      """
      Randomize the weights
      """

      self.parameters = np.random.uniform(-0.1, 0.1, self.dimensions)


   def forward(self):
      """
      Perform a forward step
      """

      self.inputs = self.from_layer.getOutput()
      self.outputs = np.dot(self.inputs, self.parameters)   # This ensures correct dimensions


   def backward(self):
      """
      Perform a backprop step
      """

      self.delta = np.dot(self.to_layer.getDelta(), self.parameters.transpose())


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = np.zeros(self.dimensions)


   def updateParameters(self, dParams):
      """
      Update the weights in the weight matrix
      """

      self.parameters += dParams


   def updateParameterGradient(self):
      """
      Update the weight matrix based on the outer tensor product of the forward
      and backward passes
      """

      self.gradient += np.tensordot(self.inputs.transpose(), self.to_layer.getDelta(), 1)


