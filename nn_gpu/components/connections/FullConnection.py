## FullConnection.py	Dana Hughes		01-Sept-2015
##

import numpy as np
import gnumpy as gpu

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
      self.parameters = gpu.zeros(self.dimensions)

      self.input = gpu.zeros((1,from_port.size))
      self.output = gpu.zeros((1,to_port.size))
      self.delta = gpu.zeros((1,from_port.size))

      self.gradient = gpu.zeros(self.dimensions)


   def randomize(self):
      """
      Randomize the weights
      """

      self.parameters = gpu.garray(np.random.uniform(-0.1, 0.1, self.dimensions))


   def forward(self):
      """
      Perform a forward step
      """

      self.input = self.from_port.getOutput()
      self.output = gpu.dot(self.input, self.parameters)   # This ensures correct dimensions


   def backward(self):
      """
      Perform a backprop step
      """

      self.delta = gpu.dot(self.to_port.getDelta(), self.parameters.transpose())


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = gpu.zeros(self.dimensions)


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

      self.gradient += gpu.tensordot(self.input.transpose(), self.to_port.getDelta(), 1)


