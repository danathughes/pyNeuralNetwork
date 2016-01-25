## FullConnection.py	Dana Hughes		01-Sept-2015
##

import numpy as np
from AbstractConnection import AbstractConnection

class FullConnection(AbstractConnection):
   """
   A connection which fully links two layers.
   """

   def __init__(self, from_port, to_port):
      """
      Create a new full connection
      """

      # Properly initialize the abstract connection
      AbstractConnection.__init__(self, from_port, to_port)

      # Determine the dimensions and initialize the weight matrix
      self.dimensions = (from_port.size, to_port.size)
      self.parameters = np.zeros(self.dimensions)

      self.input = np.zeros((1,from_port.size))
      self.output = np.zeros((1,to_port.size))
      self.delta = np.zeros((1,from_port.size))

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

      self.input = self.from_port.getOutput()
      self.output = np.dot(self.input, self.parameters)   # This ensures correct dimensions


   def backward(self):
      """
      Perform a backprop step
      """

      self.delta = np.dot(self.to_port.getDelta(), self.parameters.transpose())


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = np.zeros(self.dimensions)


   def setParameters(self, params):
      """
      Set the weights in the weight matrix
      """

      self.parameters[:] = params[:]


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

      self.gradient += np.tensordot(self.input.transpose(), self.to_port.getDelta(), 1)


