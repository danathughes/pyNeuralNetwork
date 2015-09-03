## Bias.py	Dana Hughes		02-Sept-2015
##

import numpy as np


class Bias:
   """
   A bias for use as input to a layer
   """

   def __init__(self, size):
      """
      Create a new bias
      """

      self.weights = np.zeros((1, size))
      self.dimensions = (1, size)

      self.inputs = np.zeros((0,0))
      self.outputs = np.zeros((1,size))

      self.gradient = np.zeros(self.dimensions)

      self.output_connection = None


   def randomize(self):
      """
      """

      self.weights = np.random.uniform(-0.1, 0.1, self.dimensions)


   def setInputConnection(self, layer):
      """
      Bias units do not have an input connection
      """

      pass



   def setOutputConnection(self, layer):
      """
      Connect this bias to the provided layer
      """

      self.output_connection = layer


   def forward(self):
      """
      Perform a forward step (just propagate the weights)
      """

      self.outputs = self.weights  


   def backward(self):
      """
      Biases do not need to propagate anything backward
      """

      pass


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = np.zeros(self.dimensions)


   def updateParameters(self, params):
      """
      Update the parameters
      """

      self.weights += params


   def getParameters(self):
      """
      Return the gradient of this layer
      """

      return self.weights


   def getInput(self):
      """
      Provide the input to this unit
      """

      return self.inputs


   def getOutput(self):
      """
      Provide the output from this unit
      """

      return self.outputs


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
       
