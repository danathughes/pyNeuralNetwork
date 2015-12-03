## Bias.py	Dana Hughes		02-Sept-2015
##

import numpy as np
import gnumpy as gpu

from AbstractConnection import AbstractConnection


class Bias(AbstractConnection):
   """
   A bias for use as input to a layer
   """

   def __init__(self, to_port):
      """
      Create a new bias
      """

      # Properly initialize the Bias
      AbstractConnection.__init__(self, None, to_port)

      self.parameters = gpu.zeros((1, to_port.size))
      self.dimensions = (1, to_port.size)

      self.input = gpu.zeros((0,0))
      self.output = gpu.zeros((1,to_port.size))

      self.gradient = gpu.zeros(self.dimensions)


   def randomize(self):
      """
      Randomize the bias
      """

      self.parameters = gpu.garray(np.random.uniform(-0.1, 0.1, self.dimensions))


   def forward(self):
      """
      Perform a forward step (just propagate the weights)
      """

      self.output = self.parameters  


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = gpu.zeros(self.dimensions)


   def updateParameters(self, dParams):
      """
      Update the parameters
      """

      self.parameters += dParams


   def updateParameterGradient(self):
      """
      Update the parameter gradient with the appropriate weight change based on forward and backward pass
      """

      self.gradient += gpu.sum(self.to_port.getDelta(), 0)


   def getParameterGradient(self):
      """
      Return the gradient after backpropagation
      """

      return self.gradient


   def getDelta(self):
      """
      Return the delta after a backward pass
      """

      return None
       
