## Bias.py	Dana Hughes		02-Sept-2015
##

import numpy as np
from AbstractConnection import AbstractConnection


class Bias(AbstractConnection):
   """
   A bias for use as input to a layer
   """

   def __init__(self, size):
      """
      Create a new bias
      """

      # Properly initialize the Bias
      AbstractConnection.__init__(self)

      self.parameters = np.zeros((1, size))
      self.dimensions = (1, size)

      self.input = np.zeros((0,0))
      self.output = np.zeros((1,size))

      self.gradient = np.zeros(self.dimensions)

      self.from_layer = None


   def randomize(self):
      """
      Randomize the bias
      """

      self.parameters = np.random.uniform(-0.1, 0.1, self.dimensions)


   def setFromLayer(self, layer):
      """
      Does nothing, as bias units only output to a layer
      """

      pass


   def forward(self):
      """
      Perform a forward step (just propagate the weights)
      """

      self.outputs = self.parameters  


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = np.zeros(self.dimensions)


   def updateParameters(self, dParams):
      """
      Update the parameters
      """

      self.parameters += params

   def updateParameterGradient(self):
      """
      Update the parameter gradient with the appropriate weight change based on forward and backward pass
      """

      self.gradient += np.sum(self.output_connection.getDelta(), 0)


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
       
