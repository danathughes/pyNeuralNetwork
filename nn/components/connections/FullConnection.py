## FullConnection.py	Dana Hughes		01-Sept-2015
##

import numpy as np


class FullConnection:
   """
   An input layer
   """

   def __init__(self, inputSize, outputSize):
      """
      """

      self.weights = np.zeros((inputSize, outputSize))
      self.dimensions = (inputSize, outputSize)

      self.inputs = np.zeros((0,inputSize))
      self.outputs = np.zeros((0,outputSize))

      self.gradient = np.zeros(self.dimensions)

      self.input_connection = None
      self.output_connection = None


   def randomize(self):
      """
      """

      self.weights = np.random.uniform(-0.1, 0.1, self.dimensions)


   def setInputConnection(self, layer):
      """
      """

      self.input_connection = layer



   def setOutputConnection(self, layer):
      """
      """

      self.output_connection = layer


   def forward(self):
      """
      Perform a forward step
      """

      self.inputs = self.input_connection.getOutput()
      self.outputs = np.dot(self.inputs, self.weights)   # This ensures correct dimensions


   def backward(self):
      """
      Perform a backprop step
      """

      self.delta = np.dot(self.output_connection.getDelta(), self.weights.transpose())


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

      self.gradient += np.tensordot(self.inputs.transpose(), self.output_connection.getDelta(), 1)


   def getParameterGradient(self):
      """
      Return the gradient after backpropagation
      """

      return self.gradient


   def getDelta(self):
      """
      Return the delta after a backward pass
      """

      return self.delta
       